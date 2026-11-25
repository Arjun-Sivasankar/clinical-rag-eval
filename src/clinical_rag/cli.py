"""Command line: `crag prepare`, `crag index`, `crag eval-retrieval`."""

from __future__ import annotations

import argparse
import hashlib
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


def _texts_sha1(texts: list[str]) -> str:
    """Hash of an ordered list of texts; each text is length-prefixed, so no two lists collide
    by concatenation."""
    h = hashlib.sha1()
    for t in texts:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


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


def _load_parents() -> dict[str, str]:
    df = pd.read_parquet(get_settings().data_dir / "docs.parquet", columns=["pmid", "text"])
    return dict(zip(df["pmid"], df["text"], strict=True))


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

    # BM25 length normalisation uses the corpus's own mean document length (in BM25 tokens),
    # not fastembed's default of 256. The cache is reused only for the exact same avg_len and
    # chunk texts, built by the same fastembed version (its tokenizer and stemmer).
    from importlib.metadata import version

    avg_len = args.bm25_avg_len or FastEmbedBM25Encoder(s.sparse_model).mean_doc_len(texts)
    sparse_key = {
        "avg_len": avg_len,
        "texts_sha1": _texts_sha1(texts),
        "fastembed": version("fastembed"),
    }
    sparse_path = s.data_dir / f"sparse_{_slug(s.sparse_model)}.pkl"
    cached = pickle.loads(sparse_path.read_bytes()) if sparse_path.exists() else None
    if isinstance(cached, dict) and cached.get("key") == sparse_key and not args.reembed:
        sparse = cached["vectors"]
    else:
        enc_s = FastEmbedBM25Encoder(s.sparse_model, avg_len=avg_len)
        t0 = time.perf_counter()
        sparse = enc_s.encode_docs(texts)
        timing["sparse_embed_s"] = time.perf_counter() - t0
        sparse_path.write_bytes(pickle.dumps({"key": sparse_key, "vectors": sparse}))
    timing["bm25_avg_len"] = avg_len

    # The collection is about to be replaced: drop metadata describing the old one first.
    (s.data_dir / "index_meta.json").unlink(missing_ok=True)
    client = make_client(s.qdrant_location)
    create_collection(client, s.collection, dim=dense.shape[1], recreate=True)
    t0 = time.perf_counter()
    upsert_chunks(client, s.collection, chunks, dense, sparse)
    timing["upsert_s"] = time.perf_counter() - t0
    timing["n_points"] = client.count(s.collection, exact=True).count
    # Written only after a successful upload, so eval metadata describes what is in Qdrant.
    (s.data_dir / "index_meta.json").write_text(
        json.dumps({"bm25_avg_len": avg_len, "sparse_model": s.sparse_model}, indent=2) + "\n",
        encoding="utf-8",
    )
    # Merge, so a re-upload from cached embeddings keeps the embedding timings of the first run.
    tpath = s.data_dir / f"index_timing_{_slug(s.dense_model)}.json"
    old = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    timing = {**old, **timing}
    tpath.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log.info("indexed: %s", timing)


def cmd_eval_retrieval(args) -> None:
    from .encoders import FastEmbedBM25Encoder, SentenceTransformerEncoder
    from .evaluate import evaluate_systems, format_table, write_results
    from .index import make_client
    from .rerank import CrossEncoderReranker
    from .retrieve import Retriever

    s = get_settings()
    queries = pd.read_parquet(s.data_dir / "queries.parquet")
    if args.limit_queries:
        queries = queries.head(args.limit_queries)
    stats = json.loads((s.data_dir / "corpus_stats.json").read_text(encoding="utf-8"))
    index_meta_path = s.data_dir / "index_meta.json"
    index_meta = (
        json.loads(index_meta_path.read_text(encoding="utf-8")) if index_meta_path.exists() else {}
    )
    client = make_client(s.qdrant_location)
    retriever = Retriever(
        client,
        s.collection,
        SentenceTransformerEncoder(s.dense_model, s.dense_query_prefix, s.device),
        FastEmbedBM25Encoder(s.sparse_model, avg_len=index_meta.get("bm25_avg_len")),
        prefetch_limit=args.prefetch_limit,
        reranker=CrossEncoderReranker(args.reranker, s.device) if args.reranker else None,
        rerank_depth=args.rerank_depth,
        parents=_load_parents() if "rerank_parent" in args.systems else None,
        rerank_parents=args.rerank_parents,
    )
    systems = args.systems.split(",")
    results = evaluate_systems(retriever, systems, queries, chunk_limit=args.chunk_limit)
    summaries = [summary for summary, _ in results]
    per_query = {system: rows for system, (_, rows) in zip(systems, results, strict=True)}
    # Embedded (local) mode has no server info endpoint.
    server = client.info().version if s.qdrant_location.startswith("http") else "embedded"
    meta = {
        **stats,
        "dense_model": s.dense_model,
        "sparse_model": s.sparse_model,
        "bm25_avg_len": index_meta.get("bm25_avg_len"),
        "qdrant_location": s.qdrant_location,
        "qdrant_server": server,
        "chunk_limit": args.chunk_limit,
        "prefetch_limit": args.prefetch_limit,
        "device": retriever.dense.device,
        "reranker": args.reranker or None,
        "reranker_dtype": retriever.reranker.dtype if retriever.reranker else None,
        # chunk mode reranks the top rerank_depth chunks; parent mode the top rerank_parents
        # abstracts found in the base retriever's top max(chunk_limit, prefetch_limit) chunks
        "rerank_depth": args.rerank_depth if any(x.endswith("+rerank") for x in systems) else None,
        "rerank_parents": args.rerank_parents if "rerank_parent" in args.systems else None,
    }
    path = write_results(s.results_dir, args.name, summaries, per_query, meta)
    print(format_table(summaries))
    log.info("wrote %s", path)


def cmd_generate(args) -> None:
    """Answer the questions with the RAG pipeline. Resumable: rows already in
    results/<name>.jsonl are kept and skipped."""
    from datetime import UTC, datetime

    from tqdm import tqdm

    from .encoders import FastEmbedBM25Encoder, SentenceTransformerEncoder
    from .evaluate import hardware_info
    from .generate import LLMClient
    from .index import make_client
    from .rag import RAGPipeline, summarise
    from .rerank import CrossEncoderReranker
    from .retrieve import Retriever

    s = get_settings()
    queries = pd.read_parquet(s.data_dir / "queries.parquet")
    if args.limit_queries:
        queries = queries.head(args.limit_queries)
    index_meta = json.loads((s.data_dir / "index_meta.json").read_text(encoding="utf-8"))
    retriever = Retriever(
        make_client(s.qdrant_location),
        s.collection,
        SentenceTransformerEncoder(s.dense_model, s.dense_query_prefix, s.device),
        FastEmbedBM25Encoder(s.sparse_model, avg_len=index_meta.get("bm25_avg_len")),
        reranker=CrossEncoderReranker(args.reranker, s.device)
        if "+rerank" in args.system
        else None,
        parents=_load_parents(),
    )
    llm = LLMClient(args.base_url, args.model)
    pipe = RAGPipeline(
        retriever, llm, retriever.parents, args.system, args.top_k, args.rewrite, args.max_tokens
    )
    rows_path = s.results_dir / f"{args.name}.jsonl"
    done = {}
    if rows_path.exists():
        with open(rows_path, encoding="utf-8") as f:
            done = {r["qid"]: r for r in map(json.loads, f)}
    with open(rows_path, "a", encoding="utf-8", newline="\n") as f:
        for row in tqdm(queries.itertuples(index=False), total=len(queries), desc=args.name):
            if row.qid in done:
                continue
            out = {
                "qid": row.qid,
                "gold_pmid": row.gold_pmid,
                "gold_label": row.final_decision,
                **pipe.run(row.question),
            }
            f.write(json.dumps(out) + "\n")
            f.flush()
            done[row.qid] = out
    rows = [done[q] for q in queries["qid"] if q in done]
    summary = summarise(rows)
    meta = {
        "model": args.model,
        "backend": args.backend,
        "generator_hardware": args.generator_hardware,
        "retrieval_hardware": hardware_info(),
        "system": args.system,
        "reranker": args.reranker if "+rerank" in args.system else None,
        "top_k": args.top_k,
        "rewrite": args.rewrite,
        "max_tokens": args.max_tokens,
        "bm25_avg_len": index_meta.get("bm25_avg_len"),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (s.results_dir / f"{args.name}.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log.info("%s: %s", args.name, {k: summary[k] for k in ("n", "accuracy", "macro_f1")})


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
    pi.add_argument(
        "--bm25-avg-len", type=float, default=0.0, help="BM25 avg doc length (default: corpus mean)"
    )
    pi.set_defaults(fn=cmd_index)

    pe = sub.add_parser("eval-retrieval", help="retrieval ablation with metrics and latency")
    pe.add_argument("--systems", default="bm25,dense,hybrid_rrf,hybrid_dbsf")
    pe.add_argument("--limit-queries", type=int, default=0)
    pe.add_argument("--chunk-limit", type=int, default=100)
    pe.add_argument("--prefetch-limit", type=int, default=100)
    pe.add_argument("--reranker", default="", help="cross-encoder model for `<system>+rerank`")
    pe.add_argument("--rerank-depth", type=int, default=50, help="chunks passed to the reranker")
    pe.add_argument(
        "--rerank-parents", type=int, default=20, help="abstracts passed to `+rerank_parent`"
    )
    pe.add_argument("--name", default="m1_retrieval")
    pe.set_defaults(fn=cmd_eval_retrieval)

    pg = sub.add_parser("generate", help="answer questions with the RAG pipeline (resumable)")
    pg.add_argument("--base-url", required=True, help="OpenAI-compatible API, e.g. .../v1")
    pg.add_argument("--model", required=True, help="model name as the server knows it")
    pg.add_argument("--backend", required=True, help="serving stack, e.g. 'vllm 0.x' or 'mlx-lm'")
    pg.add_argument(
        "--generator-hardware", required=True, help="where the model runs, e.g. 'Mac mini M4'"
    )
    pg.add_argument("--system", default="hybrid_dbsf+rerank_parent")
    pg.add_argument("--reranker", default="ncbi/MedCPT-Cross-Encoder")
    pg.add_argument("--top-k", type=int, default=3, help="abstracts given to the generator")
    pg.add_argument("--rewrite", action="store_true", help="rewrite the query before retrieval")
    pg.add_argument("--max-tokens", type=int, default=256)
    pg.add_argument("--limit-queries", type=int, default=0)
    pg.add_argument("--name", required=True, help="results/<name>.jsonl and .json")
    pg.set_defaults(fn=cmd_generate)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
