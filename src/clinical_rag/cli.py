"""Command line: `crag prepare`, `crag index`, `crag eval-retrieval`."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time

import numpy as np
import pandas as pd

from .config import get_settings

log = logging.getLogger("clinical_rag")


def _slug(name: str) -> str:
    return name.replace("/", "__")


def cmd_prepare(args) -> None:
    from .chunking import chunk_corpus
    from .data import build_corpus, download

    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    paths = download()
    docs, queries = build_corpus(paths)
    if args.max_docs:
        # Keep every gold abstract, then fill with distractors up to max_docs.
        gold = set(queries["gold_pmid"])
        keep = [d for d in docs if d.pmid in gold]
        if args.max_docs < len(keep):
            log.warning(
                "--max-docs %d < %d gold abstracts; keeping all gold", args.max_docs, len(keep)
            )
        keep += [d for d in docs if d.pmid not in gold][: max(0, args.max_docs - len(keep))]
        docs = keep
    chunks = chunk_corpus(docs, max_words=s.max_chunk_words)
    if not chunks:
        raise SystemExit("no chunks produced; check the downloaded splits")
    pd.DataFrame(
        {
            "pmid": [d.pmid for d in docs],
            "split": [d.split for d in docs],
            "text": [d.text for d in docs],
        }
    ).to_parquet(s.data_dir / "docs.parquet", index=False)
    pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "pmid": [c.pmid for c in chunks],
            "section": [c.section for c in chunks],
            "position": [c.position for c in chunks],
            "text": [c.text for c in chunks],
        }
    ).to_parquet(s.data_dir / "chunks.parquet", index=False)
    queries.to_parquet(s.data_dir / "queries.parquet", index=False)
    words = np.array([len(c.text.split()) for c in chunks])
    stats = {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "chunk_words_mean": float(words.mean()),
        "chunk_words_p95": float(np.percentile(words, 95)),
        "label_counts": queries["final_decision"].value_counts().to_dict(),
    }
    (s.data_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("prepared: %s", stats)


def _load_chunks():
    from .chunking import Chunk

    s = get_settings()
    df = pd.read_parquet(s.data_dir / "chunks.parquet")
    return [
        Chunk(
            chunk_id=int(r.chunk_id),
            pmid=r.pmid,
            section=r.section,
            position=int(r.position),
            text=r.text,
        )
        for r in df.itertuples(index=False)
    ]


def cmd_index(args) -> None:
    from .encoders import FastEmbedBM25Encoder, SentenceTransformerEncoder
    from .index import create_collection, make_client, upsert_chunks

    s = get_settings()
    chunks = _load_chunks()
    texts = [c.text for c in chunks]
    timing = {}

    dense_path = s.data_dir / f"dense_{_slug(s.dense_model)}.npy"
    if dense_path.exists() and not args.reembed:
        dense = np.load(dense_path)
    else:
        enc = SentenceTransformerEncoder(s.dense_model, s.dense_query_prefix, s.device)
        t0 = time.perf_counter()
        dense = enc.encode_docs(texts, batch_size=args.batch_size)
        timing["dense_embed_s"] = time.perf_counter() - t0
        timing["dense_device"] = enc.device
        np.save(dense_path, dense)

    sparse_path = s.data_dir / f"sparse_{_slug(s.sparse_model)}.pkl"
    if sparse_path.exists() and not args.reembed:
        sparse = pickle.loads(sparse_path.read_bytes())
    else:
        enc_s = FastEmbedBM25Encoder(s.sparse_model)
        t0 = time.perf_counter()
        sparse = enc_s.encode_docs(texts)
        timing["sparse_embed_s"] = time.perf_counter() - t0
        sparse_path.write_bytes(pickle.dumps(sparse))

    client = make_client(s.qdrant_location)
    create_collection(client, s.collection, dim=dense.shape[1], recreate=True)
    t0 = time.perf_counter()
    upsert_chunks(client, s.collection, chunks, dense, sparse)
    timing["upsert_s"] = time.perf_counter() - t0
    timing["n_points"] = client.count(s.collection, exact=True).count
    # Merge, so a re-upload from cached embeddings keeps the embedding timings of the first run.
    tpath = s.data_dir / f"index_timing_{_slug(s.dense_model)}.json"
    old = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    timing = {**old, **timing}
    tpath.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log.info("indexed: %s", timing)


def cmd_eval_retrieval(args) -> None:
    from .encoders import FastEmbedBM25Encoder, SentenceTransformerEncoder
    from .evaluate import evaluate_system, format_table, write_results
    from .index import make_client
    from .retrieve import Retriever

    s = get_settings()
    queries = pd.read_parquet(s.data_dir / "queries.parquet")
    if args.limit_queries:
        queries = queries.head(args.limit_queries)
    stats = json.loads((s.data_dir / "corpus_stats.json").read_text(encoding="utf-8"))
    client = make_client(s.qdrant_location)
    retriever = Retriever(
        client,
        s.collection,
        SentenceTransformerEncoder(s.dense_model, s.dense_query_prefix, s.device),
        FastEmbedBM25Encoder(s.sparse_model),
        prefetch_limit=args.prefetch_limit,
    )
    summaries, per_query = [], {}
    for system in args.systems.split(","):
        summary, rows = evaluate_system(retriever, system, queries, chunk_limit=args.chunk_limit)
        summaries.append(summary)
        per_query[system] = rows
    # Embedded (local) mode has no server info endpoint.
    server = client.info().version if s.qdrant_location.startswith("http") else "embedded"
    meta = {
        **stats,
        "dense_model": s.dense_model,
        "sparse_model": s.sparse_model,
        "qdrant_location": s.qdrant_location,
        "qdrant_server": server,
        "chunk_limit": args.chunk_limit,
        "prefetch_limit": args.prefetch_limit,
        "device": retriever.dense.device,
    }
    path = write_results(s.results_dir, args.name, summaries, per_query, meta)
    print(format_table(summaries))
    log.info("wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="crag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="download PubMedQA, build corpus, chunk")
    pp.add_argument("--max-docs", type=int, default=0, help="subset: all gold docs + distractors")
    pp.set_defaults(fn=cmd_prepare)

    pi = sub.add_parser("index", help="embed chunks (dense + BM25) and load Qdrant")
    pi.add_argument("--batch-size", type=int, default=256)
    pi.add_argument("--reembed", action="store_true")
    pi.set_defaults(fn=cmd_index)

    pe = sub.add_parser("eval-retrieval", help="retrieval ablation with metrics and latency")
    pe.add_argument("--systems", default="bm25,dense,hybrid_rrf,hybrid_dbsf")
    pe.add_argument("--limit-queries", type=int, default=0)
    pe.add_argument("--chunk-limit", type=int, default=100)
    pe.add_argument("--prefetch-limit", type=int, default=100)
    pe.add_argument("--name", default="m1_retrieval")
    pe.set_defaults(fn=cmd_eval_retrieval)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
