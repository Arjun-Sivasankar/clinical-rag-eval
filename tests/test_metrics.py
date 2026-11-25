import math

import pytest

from clinical_rag.metrics import mrr_at_k, ndcg_at_k, percentiles, recall_at_k


def test_recall_single_relevant():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, ["b"], 1) == 0.0
    assert recall_at_k(ranked, ["b"], 2) == 1.0
    assert recall_at_k(ranked, ["z"], 3) == 0.0


def test_recall_multiple_relevant():
    assert recall_at_k(["a", "b", "c", "d"], ["a", "d"], 3) == 0.5


def test_mrr():
    assert mrr_at_k(["a", "b", "c"], ["c"], 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["a", "b", "c"], ["c"], 2) == 0.0


def test_ndcg_single_relevant():
    assert ndcg_at_k(["x"], ["x"], 10) == 1.0
    assert ndcg_at_k(["a", "x"], ["x"], 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a", "b"], ["x"], 10) == 0.0


def test_ndcg_two_relevant_perfect():
    assert ndcg_at_k(["x", "y", "z"], ["x", "y"], 10) == pytest.approx(1.0)


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5)
    assert p["p95"] == pytest.approx(95.05)
