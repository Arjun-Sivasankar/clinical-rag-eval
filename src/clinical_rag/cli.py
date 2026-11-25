"""Command line: `crag prepare` (more commands follow with the retrieval milestone)."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from .config import get_settings

log = logging.getLogger("clinical_rag")


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="crag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="download PubMedQA, build corpus, chunk")
    pp.add_argument("--max-docs", type=int, default=0, help="subset: all gold docs + distractors")
    pp.set_defaults(fn=cmd_prepare)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
