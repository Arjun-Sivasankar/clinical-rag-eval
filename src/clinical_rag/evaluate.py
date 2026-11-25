"""Retrieval ablation: run each system over the labelled questions one query at a time (online
setting, batch size 1), score at abstract (PMID) level, and record per-stage latency."""

from __future__ import annotations

import json
import logging
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import mrr_at_k, ndcg_at_k, percentiles, recall_at_k
from .retrieve import Retriever

log = logging.getLogger(__name__)

KS = (1, 5, 10, 20)


def evaluate_system(
    retriever: Retriever,
    system: str,
    queries: pd.DataFrame,
    chunk_limit: int = 100,
    warmup: int = 10,
) -> tuple[dict, list[dict]]:
    # Warm-up queries (model load, CUDA kernels, Qdrant caches) are not timed or scored.
    for q in queries["question"].head(warmup):
        retriever.search(system, q, limit=chunk_limit)

    scores: dict[str, list[float]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    per_query = []
    for row in tqdm(queries.itertuples(index=False), total=len(queries), desc=system):
        res = retriever.search(system, row.question, limit=chunk_limit)
        ranked = res.ranked_pmids()
        gold = [row.gold_pmid]
        for k in KS:
            scores[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        scores["mrr@10"].append(mrr_at_k(ranked, gold, 10))
        scores["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
        for stage, ms in res.timings_ms.items():
            timings[stage].append(ms)
        rank = ranked.index(row.gold_pmid) + 1 if row.gold_pmid in ranked else None
        per_query.append({"qid": row.qid, "gold_rank": rank, "top10": ranked[:10]})

    summary = {"system": system, "n_queries": len(queries)}
    summary.update({m: float(np.mean(v)) for m, v in scores.items()})
    summary["latency_ms"] = {stage: percentiles(v) for stage, v in timings.items()}
    return summary, per_query


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:  # pragma: no cover
        pass
    from importlib.metadata import PackageNotFoundError, version

    for pkg in ("qdrant-client", "sentence-transformers", "fastembed"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover
            pass
    return info


def format_table(summaries: list[dict]) -> str:
    has_rerank = any("rerank" in s["latency_ms"] for s in summaries)
    cols = [
        "System",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "MRR@10",
        "nDCG@10",
        "Search p50 / p95 (ms)",
    ]
    if has_rerank:
        cols.append("Rerank p50 / p95 (ms)")
    cols.append("End-to-end p50 / p95 (ms)")
    head = "| " + " | ".join(cols) + " |\n" + "|---" * len(cols) + "|\n"

    def pair(lat: dict, stage: str) -> str:
        return f"{lat[stage]['p50']:.1f} / {lat[stage]['p95']:.1f}" if stage in lat else "-"

    rows = []
    for s in summaries:
        lat = s["latency_ms"]
        cells = [s["system"]] + [
            f"{s[m]:.3f}"
            for m in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10")
        ]
        cells.append(pair(lat, "search"))
        if has_rerank:
            cells.append(pair(lat, "rerank"))
        cells.append(pair(lat, "total"))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def write_results(
    out_dir: Path, name: str, summaries: list[dict], per_query: dict[str, list[dict]], meta: dict
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        **meta,
        "hardware": hardware_info(),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / f"{name}.json").write_text(
        json.dumps({"meta": meta, "summaries": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    runs = out_dir / "runs"
    runs.mkdir(exist_ok=True)
    for system, rows in per_query.items():
        with open(runs / f"{name}_{system}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    path = out_dir / f"{name}.md"
    path.write_text(render_markdown(name, summaries, meta), encoding="utf-8", newline="\n")
    return path


def render_markdown(name: str, summaries: list[dict], meta: dict) -> str:
    """Markdown report for one results file; `python -m clinical_rag.evaluate <json>` re-renders
    it from the JSON."""
    hw = meta["hardware"]
    rerank = bool(meta.get("reranker"))
    return (
        f"# {name}\n\n"
        f"Corpus: {meta.get('n_docs')} abstracts, {meta.get('n_chunks')} chunks. "
        f"Queries: {summaries[0]['n_queries']} PubMedQA expert-labelled questions. "
        f"Dense model: `{meta.get('dense_model')}`; sparse: `{meta.get('sparse_model')}`"
        + (
            f"; reranker: `{meta['reranker']}` (`+rerank`: top {meta.get('rerank_depth')} chunks; "
            f"`+rerank_parent`: top {meta.get('rerank_parents')} parent abstracts)"
            if rerank
            else ""
        )
        + ". "
        f"Hardware: {hw.get('gpu', 'CPU only')} ({hw.get('gpu_vram_gb', '-')} GB), "
        f"{hw.get('platform')}; "
        f"Qdrant {meta.get('qdrant_server', '')} at `{meta.get('qdrant_location')}`.\n\n"
        "Scored at abstract level (best chunk per PMID) over the top "
        f"{meta.get('chunk_limit')} retrieved chunks; one gold abstract per question, so "
        "Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call"
        + ("; *rerank* is the batched cross-encoder pass" if rerank else "")
        + "; *end-to-end* adds query encoding.\n\n"
        + format_table(summaries)
        + f"\nGenerated {meta['finished_utc']} by `crag eval-retrieval`.\n"
    )


def _rerender(json_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    md = render_markdown(json_path.stem, data["summaries"], data["meta"])
    json_path.with_suffix(".md").write_text(md, encoding="utf-8", newline="\n")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        _rerender(Path(arg))
