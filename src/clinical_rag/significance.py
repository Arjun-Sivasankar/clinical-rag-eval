"""Paired significance for retrieval differences, computed from the per-query run files.

Paired bootstrap over queries (95% CI of the difference) for Recall@1, MRR@10 and nDCG@10, plus
an exact McNemar test on Recall@1 (which questions each system gets right at rank 1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_ranks(path: Path) -> dict[str, int | None]:
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r["gold_rank"] for r in map(json.loads, f)}


def per_query(ranks: dict[str, int | None], qids: list[str]) -> dict[str, np.ndarray]:
    r = np.array([ranks[q] if ranks[q] is not None else 10**9 for q in qids], dtype=float)
    return {
        "recall@1": (r <= 1).astype(float),
        "mrr@10": np.where(r <= 10, 1.0 / r, 0.0),
        "ndcg@10": np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0),
    }


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar p-value on binary outcomes. Returns (b_only, a_only, p)."""
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n, k = a_only + b_only, min(a_only, b_only)
    if n == 0:
        return b_only, a_only, 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b_only, a_only, min(1.0, 2 * p)


def compare(base_run: Path, new_run: Path, n_boot: int = 10_000, seed: int = 0) -> dict:
    base, new = load_ranks(base_run), load_ranks(new_run)
    qids = sorted(base)
    assert set(qids) == set(new), "runs must cover the same queries"
    mb, mn = per_query(base, qids), per_query(new, qids)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    out = {"base": base_run.stem, "new": new_run.stem, "n_queries": len(qids)}
    for m in mb:
        diff = mn[m] - mb[m]
        boots = diff[idx].mean(axis=1)
        out[m] = {
            "base": float(mb[m].mean()),
            "new": float(mn[m].mean()),
            "diff": float(diff.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    gained, lost, p = mcnemar_exact(mb["recall@1"], mn["recall@1"])
    out["mcnemar_recall@1"] = {"gained": gained, "lost": lost, "p_value": p}
    return out


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone), in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m, running, adjusted = len(p_values), 0.0, [0.0] * len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def holm_family(paths: list[Path]) -> list[dict]:
    """Treat every comparison in the given significance files as one family and add Holm-adjusted
    McNemar p-values."""
    rows = [r for p in paths for r in json.loads(p.read_text(encoding="utf-8"))]
    adj = holm([r["mcnemar_recall@1"]["p_value"] for r in rows])
    return [
        {
            "base": r["base"],
            "new": r["new"],
            "recall@1_diff": r["recall@1"]["diff"],
            "p_value": r["mcnemar_recall@1"]["p_value"],
            "p_holm": a,
        }
        for r, a in zip(rows, adj, strict=True)
    ]


def main(argv: list[str] | None = None) -> None:
    """`significance <base> <new>...` compares runs; `significance holm <files>...` adjusts a family
    of saved comparisons for multiple testing."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__)
    if argv and argv[0] == "holm":
        ap.add_argument("files", nargs="+", type=Path)
        ap.add_argument("--out", type=Path)
        args = ap.parse_args(argv[1:])
        results = holm_family(args.files)
    else:
        ap.add_argument("base", type=Path)
        ap.add_argument("new", nargs="+", type=Path)
        ap.add_argument("--out", type=Path)
        args = ap.parse_args(argv)
        results = [compare(args.base, n) for n in args.new]
    text = json.dumps(results, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
