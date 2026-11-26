"""Static results page for GitHub Pages: `python -m clinical_rag.site` -> site/index.html.

Everything on the page is read from `results/` (the same rows as the README table, plus the
Holm-adjusted significance files), so the page cannot drift from the runs. Charts are inline SVG
dot plots with native hover tooltips; the full table is the accessible view of the same numbers.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from .config import REPO_ROOT, get_settings
from .report import collect

REPO_URL = "https://github.com/Arjun-Sivasankar/clinical-rag-eval"

CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--mark:#2a78d6;
--font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#383835;--mark:#3987e5}}
body{margin:0;background:var(--surface);color:var(--ink);font:16px/1.5 var(--font)}
main{max-width:980px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px}
p,li{color:var(--ink)}.muted{color:var(--ink2);font-size:14px}
.hero{font-size:40px;font-weight:650;margin:16px 0 0}
table{border-collapse:collapse;width:100%;font-size:14px;margin-top:8px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
svg text{fill:var(--ink2);font:12px var(--font)}svg .label{fill:var(--ink)}
svg .grid{stroke:var(--grid);stroke-width:1}svg .dot{fill:var(--mark)}
svg .hit{fill:transparent}svg g.row:hover .dot{stroke:var(--ink);stroke-width:2}
a{color:var(--mark)}
"""


def dot_plot(
    rows: list[dict], value, fmt: str, lo: float, hi: float, ticks: list[float], unit: str = ""
) -> str:
    """Horizontal dot plot, one row per pipeline, one shared axis from lo to hi."""
    label_w, plot_w, row_h, top = 330, 560, 26, 24
    width, height = label_w + plot_w + 20, top + row_h * len(rows) + 28

    def x(v: float) -> float:  # clamped, so an out-of-range value never covers the labels
        return label_w + min(max((v - lo) / (hi - lo), 0.0), 1.0) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for t in ticks:
        parts.append(
            f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 8}" '
            f'y2="{height - 24}"/><text x="{x(t):.1f}" y="{height - 8}" '
            f'text-anchor="middle">{format(t, fmt)}{unit}</text>'
        )
    for i, r in enumerate(rows):
        y = top + i * row_h + row_h / 2
        v = value(r)
        tip = html.escape(f"{r['label']}: {format(v, fmt)}{unit}")
        parts.append(
            f'<g class="row"><title>{tip}</title>'
            f'<rect class="hit" x="0" y="{y - row_h / 2:.1f}" width="{width}" height="{row_h}"/>'
            f'<text class="label" x="{label_w - 12}" y="{y + 4:.1f}" text-anchor="end">'
            f"{html.escape(r['label'])}</text>"
            f'<circle class="dot" cx="{x(v):.1f}" cy="{y:.1f}" r="5"/></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _table(rows: list[dict]) -> str:
    head = (
        "<tr><th>Pipeline</th><th>Recall@1</th><th>Recall@5</th><th>MRR@10</th>"
        "<th>nDCG@10</th><th>Rerank p95 (ms)</th><th>End-to-end p95 (ms)</th></tr>"
    )
    body = []
    for r in rows:
        lat = r["latency_ms"]
        rerank = f"{lat['rerank']['p95']:.0f}" if "rerank" in lat else "-"
        body.append(
            f"<tr><td>{html.escape(r['label'])}</td><td>{r['recall@1']:.3f}</td>"
            f"<td>{r['recall@5']:.3f}</td><td>{r['mrr@10']:.3f}</td><td>{r['ndcg@10']:.3f}</td>"
            f"<td>{rerank}</td><td>{lat['total']['p95']:.0f}</td></tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


_SYSTEMS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_dbsf": "Hybrid DBSF",
}
_RERANKERS = {
    "MedCPT-Cross-Encoder": "MedCPT",
    "ms-marco-MiniLM-L6-v2": "MiniLM-L6",
    "bge-reranker-base": "bge-reranker-base",
}


def pretty_run(name: str) -> str:
    """'m2_rerank_MedCPT-Cross-Encoder_hybrid_dbsf+rerank_parent' -> 'Hybrid DBSF + MedCPT
    (abstracts)'; unknown names are returned unchanged."""
    reranker = ""
    for prefix in ("m1_retrieval_", "m2_rerank_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for key, short in _RERANKERS.items():
        if name.startswith(key + "_"):
            reranker, name = short, name[len(key) + 1 :]
    base, _, stage = name.partition("+")
    label = _SYSTEMS.get(base, base)
    if stage and reranker:
        label += f" + {reranker} ({'abstracts' if stage == 'rerank_parent' else 'chunks'})"
    return label


def _significance(results_dir: Path) -> str:
    items = []
    for name in ("m1_significance_holm.json", "m2_significance_holm.json"):
        path = results_dir / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            items.append(
                f"<tr><td>{html.escape(pretty_run(r['new']))} vs "
                f"{html.escape(pretty_run(r['base']))}</td>"
                f"<td>{r['recall@1_diff'] * 100:+.1f}</td><td>{r['p_value']:.2g}</td>"
                f"<td>{r['p_holm']:.2g}</td></tr>"
            )
    if not items:
        return ""
    return (
        "<h2>Is it significant?</h2><p class='muted'>Paired exact McNemar tests on Recall@1 over "
        "the per-query runs; Holm-adjusted within each family of comparisons.</p>"
        "<table><tr><th>Comparison</th><th>Recall@1 diff (points)</th>"
        f"<th>p</th><th>Holm p</th></tr>{''.join(items)}</table>"
    )


def build_page(results_dir: Path) -> str:
    rows, hw = collect(results_dir)
    best = max(rows, key=lambda r: r["recall@1"])
    bm25 = next((r for r in rows if r["system"] == "bm25"), None)
    n_queries = rows[0]["n_queries"]
    corpus_path = results_dir / "m1_corpus_and_indexing.json"
    n_docs = (
        json.loads(corpus_path.read_text(encoding="utf-8"))["corpus"]["n_docs"]
        if corpus_path.exists()
        else None
    )
    hardware = f"{hw.get('gpu', 'CPU')} ({hw.get('gpu_vram_gb', '?')} GB), {hw.get('platform')}"
    # Recall axis: from the lowest tenth below the data (at most 0.6) up to 1.0.
    r_lo = min(0.6, math.floor(min(r["recall@1"] for r in rows) * 10) / 10)
    recall_plot = dot_plot(
        rows,
        lambda r: r["recall@1"],
        ".1f",
        r_lo,
        1.0,
        [round(r_lo + i / 10, 1) for i in range(round((1.0 - r_lo) * 10) + 1)],
    )
    p95 = [r["latency_ms"]["total"]["p95"] for r in rows]
    top = max(300.0, math.ceil(max(p95) / 100) * 100)  # multiple of 100: exact quarter ticks
    latency_plot = dot_plot(
        rows,
        lambda r: r["latency_ms"]["total"]["p95"],
        ".0f",
        0,
        top,
        [0, top / 4, top / 2, 3 * top / 4, top],
        " ms",
    )
    vs_bm25 = f", up from {bm25['recall@1']:.3f} for BM25," if bm25 else ""
    corpus = f" ({n_docs:,}-abstract corpus)" if n_docs else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>clinical-rag-eval results</title><style>{CSS}</style></head>
<body><main>
<h1>clinical-rag-eval</h1>
<p class="muted">Biomedical RAG on PubMedQA, measured stage by stage on one laptop GPU.
<a href="{REPO_URL}">Code and data</a> ·
<a href="{REPO_URL}/releases">Releases</a></p>
<p class="hero">{best["recall@1"]:.3f} Recall@1</p>
<p>{html.escape(best["label"])}{vs_bm25} at
{best["latency_ms"]["total"]["p95"]:.0f} ms p95 end to end.</p>
<h2>Retrieval quality: Recall@1</h2>
<p class="muted">Share of the {n_queries:,} questions whose gold abstract is ranked
first{corpus}. Hover a row for its value.</p>
{recall_plot}
<h2>Latency: end-to-end p95</h2>
<p class="muted">One query at a time: query encoding + Qdrant search + reranking. Systems in one
results file run interleaved; compare within a run.</p>
{latency_plot}
<h2>All numbers</h2>
{_table(rows)}
{_significance(results_dir)}
<h2>Setup</h2>
<p class="muted">Hardware: {html.escape(hardware)}; Qdrant 1.20.0 in Docker on the same laptop;
encoders and rerankers in fp16. Data: PubMedQA (MIT licence), 1,000 expert-labelled questions.
Generation, RAGAS and the validated LLM judge are added as they are run (see the
<a href="{REPO_URL}/milestones">milestones</a>).</p>
<p class="muted">Generated by <code>python -m clinical_rag.site</code>
from <code>results/</code>.</p>
</main></body></html>
"""


def main() -> None:
    out = REPO_ROOT / "site" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_page(get_settings().results_dir), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
