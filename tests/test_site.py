"""Results page built from a tiny fixture of result files (no models, no Qdrant)."""

import json
import re

import pytest

from clinical_rag.report import ROWS, build, collect
from clinical_rag.site import build_page, dot_plot, pretty_run


def _summary(system, r1, p95, rerank=False):
    lat = {"search": {"p50": 10.0, "p95": 20.0}, "total": {"p50": p95 / 2, "p95": p95}}
    if rerank:
        lat["rerank"] = {"p50": 50.0, "p95": 80.0}
    return {
        "system": system,
        "n_queries": 3,
        "recall@1": r1,
        "recall@5": 0.9,
        "recall@10": 0.95,
        "recall@20": 0.97,
        "mrr@10": 0.8,
        "ndcg@10": 0.85,
        "latency_ms": lat,
    }


@pytest.fixture
def results(tmp_path):
    stems: dict[str, list] = {}
    for i, (_, stem, system) in enumerate(ROWS):
        stems.setdefault(stem, []).append(
            _summary(system, 0.70 + i * 0.01, 30.0 + 10 * i, "+rerank" in system)
        )
    hw = {"gpu": "Test GPU <8GB>", "gpu_vram_gb": 8.0, "platform": "TestOS"}
    for stem, summaries in stems.items():
        (tmp_path / f"{stem}.json").write_text(
            json.dumps({"meta": {"hardware": hw}, "summaries": summaries})
        )
    (tmp_path / "m1_significance_holm.json").write_text(
        json.dumps(
            [
                {
                    "base": "m1_retrieval_bm25",
                    "new": "m1_retrieval_dense",
                    "recall@1_diff": 0.029,
                    "p_value": 0.021,
                    "p_holm": 0.086,
                }
            ]
        )
    )
    return tmp_path


def test_collect_matches_rows(results):
    rows, hw = collect(results)
    assert [r["label"] for r in rows] == [label for label, _, _ in ROWS]
    assert hw["gpu"] == "Test GPU <8GB>"


def test_page_numbers_come_from_results(results):
    page = build_page(results)
    best = 0.70 + (len(ROWS) - 1) * 0.01
    assert f"{best:.3f} Recall@1" in page  # hero = best row
    assert page.count("<tr><td>") == len(ROWS) + 1  # table rows + 1 significance row
    assert "Dense vs BM25" in page and "+2.9" in page and "0.086" in page
    assert "Test GPU &lt;8GB&gt;" in page  # escaped
    assert page.count('class="dot"') == 2 * len(ROWS)  # two dot plots


def test_page_without_significance_files(results):
    (results / "m1_significance_holm.json").unlink()
    assert "Is it significant?" not in build_page(results)


def test_dot_plot_positions_within_axis():
    rows = [{"label": "a", "v": 0.6}, {"label": "b", "v": 0.95}]
    svg = dot_plot(rows, lambda r: r["v"], ".2f", 0.6, 0.95, [0.6, 0.95])
    xs = [float(x) for x in re.findall(r'cx="([\d.]+)"', svg)]
    assert xs == [330.0, 890.0]  # label column 330 px, plot 560 px wide
    assert "<title>a: 0.60</title>" in svg


def test_pretty_run():
    assert pretty_run("m1_retrieval_hybrid_dbsf") == "Hybrid DBSF"
    assert pretty_run("m2_rerank_MedCPT-Cross-Encoder_hybrid_dbsf+rerank_parent") == (
        "Hybrid DBSF + MedCPT (abstracts)"
    )
    assert pretty_run("m2_rerank_bge-reranker-base_hybrid_dbsf+rerank") == (
        "Hybrid DBSF + bge-reranker-base (chunks)"
    )
    assert pretty_run("something_else") == "something_else"


def test_markdown_table_uses_the_same_rows(results):
    md = build(results)
    data_rows = [ln for ln in md.splitlines()[2:] if ln.startswith("| ")]  # after header + |---|
    assert len(data_rows) == len(ROWS)
