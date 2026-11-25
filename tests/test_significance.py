import json

import numpy as np
import pytest

from clinical_rag.significance import compare, mcnemar_exact, per_query


def test_per_query_matches_metric_definitions():
    m = per_query({"a": 1, "b": 3, "c": None}, ["a", "b", "c"])
    assert m["recall@1"].tolist() == [1, 0, 0]
    assert m["mrr@10"].tolist() == pytest.approx([1, 1 / 3, 0])
    assert m["ndcg@10"].tolist() == pytest.approx([1, 0.5, 0])


def test_mcnemar_exact():
    a = np.array([1, 1, 1, 0, 0])
    b = np.array([1, 0, 0, 0, 1])
    gained, lost, p = mcnemar_exact(a, b)
    assert (gained, lost) == (1, 2)
    assert p == pytest.approx(1.0)  # 2 * P(X <= 1 | n=3) = 2 * 4/8, capped at 1
    assert mcnemar_exact(np.ones(3), np.ones(3))[2] == 1.0


def test_compare(tmp_path):
    def run(path, ranks):
        path.write_text(
            "".join(json.dumps({"qid": str(i), "gold_rank": r}) + "\n" for i, r in enumerate(ranks))
        )
        return path

    base = run(tmp_path / "base.jsonl", [2, 2, 1, None])
    new = run(tmp_path / "new.jsonl", [1, 1, 1, 1])
    out = compare(base, new, n_boot=200)
    assert out["recall@1"]["diff"] == pytest.approx(0.75)
    assert out["mcnemar_recall@1"]["gained"] == 3


def test_report_splice_replaces_only_the_marked_block():
    from clinical_rag.report import END, START, splice

    readme = f"# T\n\nintro\n{START}\nold table\n{END}\n\nrest\n"
    out = splice(readme, "| new |\n")
    assert out == f"# T\n\nintro\n{START}\n| new |\n{END}\n\nrest\n"
    assert splice(out, "| new |\n") == out  # idempotent
