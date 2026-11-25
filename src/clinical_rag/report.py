"""Build the combined ablation table (results/summary.md) from the result JSON files, so the README
table is generated from runs, never typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

from .config import get_settings

# (label, result file stem, system name inside that file)
ROWS = [
    ("BM25 (Qdrant sparse, IDF)", "m1_retrieval", "bm25"),
    ("Dense (bge-small-en-v1.5)", "m1_retrieval", "dense"),
    ("Hybrid, RRF fusion", "m1_retrieval", "hybrid_rrf"),
    ("Hybrid, DBSF fusion", "m1_retrieval", "hybrid_dbsf"),
    (
        "Hybrid DBSF + bge-reranker-base, chunks",
        "m2_rerank_bge-reranker-base",
        "hybrid_dbsf+rerank",
    ),
    (
        "Hybrid DBSF + bge-reranker-base, abstracts",
        "m2_rerank_bge-reranker-base",
        "hybrid_dbsf+rerank_parent",
    ),
    (
        "Hybrid DBSF + MiniLM-L6 (ms-marco), chunks",
        "m2_rerank_ms-marco-MiniLM-L6-v2",
        "hybrid_dbsf+rerank",
    ),
    (
        "Hybrid DBSF + MiniLM-L6 (ms-marco), abstracts",
        "m2_rerank_ms-marco-MiniLM-L6-v2",
        "hybrid_dbsf+rerank_parent",
    ),
    (
        "Hybrid DBSF + MedCPT cross-encoder, chunks",
        "m2_rerank_MedCPT-Cross-Encoder",
        "hybrid_dbsf+rerank",
    ),
    (
        "Hybrid DBSF + MedCPT cross-encoder, abstracts",
        "m2_rerank_MedCPT-Cross-Encoder",
        "hybrid_dbsf+rerank_parent",
    ),
]


def _pair(lat: dict, stage: str) -> str:
    return f"{lat[stage]['p50']:.0f} / {lat[stage]['p95']:.0f}" if stage in lat else "-"


def build(results_dir: Path) -> str:
    cache: dict[str, dict] = {}
    lines = [
        "| Pipeline | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 "
        "| Rerank p50 / p95 (ms) | End-to-end p50 / p95 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    hw = None
    for label, stem, system in ROWS:
        path = results_dir / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the evaluation that writes it")
        if stem not in cache:
            cache[stem] = json.loads(path.read_text(encoding="utf-8"))
        data = cache[stem]
        hw = hw or data["meta"]["hardware"]
        s = next(x for x in data["summaries"] if x["system"] == system)
        lat = s["latency_ms"]
        lines.append(
            f"| {label} | {s['recall@1']:.3f} | {s['recall@5']:.3f} | {s['recall@10']:.3f} "
            f"| {s['mrr@10']:.3f} | {s['ndcg@10']:.3f} | {_pair(lat, 'rerank')} "
            f"| {_pair(lat, 'total')} |"
        )
    hw_line = f"Hardware: {hw.get('gpu', 'CPU')}, {hw.get('platform')}." if hw else ""
    return "\n".join(lines) + "\n\n" + hw_line + "\n"


START, END = "<!-- results:start -->", "<!-- results:end -->"


def splice(readme: str, table: str) -> str:
    """Replace the block between the results markers in the README with `table`."""
    a, b = readme.index(START) + len(START), readme.index(END)
    return readme[:a] + "\n" + table.rstrip("\n") + "\n" + readme[b:]


def main() -> None:
    from .config import REPO_ROOT

    s = get_settings()
    text = build(s.results_dir)
    (s.results_dir / "summary.md").write_text(text, encoding="utf-8", newline="\n")
    readme = REPO_ROOT / "README.md"
    if readme.exists() and START in (content := readme.read_text(encoding="utf-8")):
        readme.write_text(splice(content, text), encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
