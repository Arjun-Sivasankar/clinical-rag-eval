"""LLM judge for yes/no/maybe answers, validated against PubMedQA's expert labels before use.

Validation: the judge reads a question and PubMedQA's reference conclusion (`long_answer`) and
says whether that text answers yes, no or maybe. Its agreement with the experts' `final_decision`
(accuracy, Cohen's kappa, confusion matrix) shows how far its verdicts can be trusted.

Use: the same judge reads a question and a generated answer and gives the label the answer
commits to, so free-text answers (or ones the regex parser cannot read) can be scored against
the gold label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .generate import LABELS, LLMClient, parse_label

JUDGE_SYSTEM = (
    "You grade answers to biomedical research questions. Read the question and the text, and "
    "decide what the text concludes about the question: yes, no, or maybe (the evidence is "
    "mixed, conditional or insufficient). Judge only what the text says, not what you believe. "
    "Reply with exactly one line: Answer: yes | no | maybe"
)


def judge_messages(question: str, text: str) -> list[dict]:
    user = f"Question: {question}\n\nText: {text}\n/no_think"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge(llm: LLMClient, question: str, text: str) -> tuple[str | None, str, float]:
    """(label or None, raw reply, latency in ms)."""
    c = llm.chat(judge_messages(question, text), max_tokens=16)
    return parse_label(c.text), c.text, c.latency_ms


def confusion(gold: Sequence[str], pred: Sequence[str | None]) -> dict[str, dict[str, int]]:
    """confusion[gold][pred] counts; unparsed predictions are counted under "none"."""
    cols = [*LABELS, "none"]
    m = {g: dict.fromkeys(cols, 0) for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        m[g][p if p in LABELS else "none"] += 1
    return m


def cohen_kappa(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    """Cohen's kappa over the three labels; an unparsed prediction counts as a disagreement."""
    n = len(gold)
    if n == 0:
        return float("nan")
    po = sum(g == p for g, p in zip(gold, pred, strict=True)) / n
    pe = sum((list(gold).count(lab) / n) * (list(pred).count(lab) / n) for lab in LABELS)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def agreement(gold: Sequence[str], pred: Sequence[str | None]) -> dict:
    return {
        "n": len(gold),
        "accuracy": float(np.mean([g == p for g, p in zip(gold, pred, strict=True)])),
        "cohen_kappa": cohen_kappa(gold, pred),
        "unparsed": sum(p not in LABELS for p in pred),
        "confusion": confusion(gold, pred),
    }
