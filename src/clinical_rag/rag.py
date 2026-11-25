"""End-to-end RAG: (optional) query rewrite -> retrieve + rerank -> answer, with per-stage timings.

Answers are scored against PubMedQA's gold yes/no/maybe labels (accuracy and macro-F1, the
dataset's standard metrics); retrieval for the rewritten query is scored like M1/M2.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generate import (
    LABELS,
    LLMClient,
    answer_messages,
    cited_pmids,
    parse_explanation,
    parse_label,
    parse_rewrite,
    rewrite_messages,
)
from .metrics import mrr_at_k, ndcg_at_k, percentiles, recall_at_k
from .retrieve import Retriever


@dataclass
class RAGPipeline:
    retriever: Retriever
    llm: LLMClient
    parents: Mapping[str, str]
    system: str = "hybrid_dbsf+rerank_parent"
    top_k: int = 3
    rewrite: bool = False
    max_tokens: int = 256

    def run(self, question: str) -> dict:
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        query = question
        if self.rewrite:
            c = self.llm.chat(rewrite_messages(question), max_tokens=64)
            query = parse_rewrite(c.text, fallback=question)
            timings["rewrite"] = c.latency_ms
        res = self.retriever.search(self.system, query, limit=100)
        for stage, ms in res.timings_ms.items():
            timings[f"retrieve_{stage}"] = ms
        ranked = res.ranked_pmids()
        contexts = [(pmid, self.parents[pmid]) for pmid in ranked[: self.top_k]]
        c = self.llm.chat(answer_messages(question, contexts), max_tokens=self.max_tokens)
        timings["generate"] = c.latency_ms
        timings["total"] = (time.perf_counter() - t_all) * 1000.0
        return {
            "query": query,
            "ranked": ranked[:20],
            "contexts": [p for p, _ in contexts],
            "raw": c.text,
            "label": parse_label(c.text),
            "explanation": parse_explanation(c.text),
            "cited": cited_pmids(c.text),
            "completion_tokens": c.completion_tokens,
            "timings_ms": timings,
        }


def macro_f1(gold: list[str], pred: list[str | None], labels=LABELS) -> float:
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def summarise(rows: list[dict]) -> dict:
    """Answer accuracy/macro-F1 vs gold labels, retrieval of the (rewritten) query, latency."""
    gold = [r["gold_label"] for r in rows]
    pred = [r["label"] for r in rows]
    out = {
        "n": len(rows),
        "accuracy": float(np.mean([g == p for g, p in zip(gold, pred, strict=True)])),
        "macro_f1": macro_f1(gold, pred),
        "unparsed": sum(p is None for p in pred),
        "empty_replies": sum(not r["raw"].strip() for r in rows),
        "pred_counts": {lab: pred.count(lab) for lab in LABELS},
        "gold_in_context": float(np.mean([r["gold_pmid"] in r["contexts"] for r in rows])),
        "cites_gold": float(np.mean([r["gold_pmid"] in r["cited"] for r in rows])),
    }
    for k in (1, 5, 10):
        out[f"recall@{k}"] = float(
            np.mean([recall_at_k(r["ranked"], [r["gold_pmid"]], k) for r in rows])
        )
    out["mrr@10"] = float(np.mean([mrr_at_k(r["ranked"], [r["gold_pmid"]], 10) for r in rows]))
    out["ndcg@10"] = float(np.mean([ndcg_at_k(r["ranked"], [r["gold_pmid"]], 10) for r in rows]))
    stages = sorted({s for r in rows for s in r["timings_ms"]})
    out["latency_ms"] = {
        s: percentiles([r["timings_ms"][s] for r in rows if s in r["timings_ms"]]) for s in stages
    }
    return out


def open_run(rows_path: Path, config: dict) -> dict[str, dict]:
    """Prepare a resumable run; return the rows already done, keyed by qid.

    The run's config is stored next to the rows (`<name>.config.json`); resuming with a different
    config is refused, so rows from different settings never mix. A partial last line (a run cut
    off mid-write) is dropped and the file rewritten, so the next append starts on a clean line.
    """
    cfg_path = rows_path.with_suffix(".config.json")
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != config:
            diff = {
                k: (old.get(k), config.get(k))
                for k in config.keys() | old.keys()
                if old.get(k) != config.get(k)
            }
            raise SystemExit(
                f"{rows_path.name} was made with other settings {diff}; use a new --name"
            )
    else:
        cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    done: dict[str, dict] = {}
    if not rows_path.exists():
        return done
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    good = []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # cut off mid-write: drop it
            raise
        good.append(line)
        done[row["qid"]] = row
    rows_path.write_text("".join(g + "\n" for g in good), encoding="utf-8", newline="\n")
    return done
