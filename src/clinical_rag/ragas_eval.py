"""RAGAS scores for generated answers: faithfulness, context recall and answer correctness.

Uses RAGAS 0.4 metric classes (`ragas.metrics.collections`), each scored per sample with the
evaluator LLM behind any OpenAI-compatible endpoint (vLLM, MLX-LM), the same servers the
generator uses. Install with `pip install -e ".[eval]"`; RAGAS is imported lazily, so the rest
of the package (and CI) does not need it.

- faithfulness: share of the answer's statements supported by the retrieved abstracts
- context recall: share of the reference conclusion's statements attributable to the abstracts
- answer correctness: 0.75 x statement-level F1 against the reference conclusion
  + 0.25 x embedding similarity (RAGAS defaults)

The reference is PubMedQA's `long_answer`, the conclusion of the gold abstract (which the
corpus does not contain).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence

from .generate import strip_think

METRICS = ("faithfulness", "context_recall", "answer_correctness")


def samples_from_generations(
    rows: Sequence[dict],
    questions: Mapping[str, str],
    references: Mapping[str, str],
    parents: Mapping[str, str],
) -> list[dict]:
    """RAGAS inputs from `crag generate` rows: the original question (not a rewritten query),
    the answer text, the abstracts the generator saw, and the reference conclusion. Rows whose
    reply is empty are skipped."""
    samples = []
    for r in rows:
        response = strip_think(r["raw"])
        if not response:
            continue
        missing = [p for p in r["contexts"] if p not in parents]
        if missing:
            raise KeyError(f"question {r['qid']}: abstracts {missing} not in docs.parquet")
        samples.append(
            {
                "qid": r["qid"],
                "user_input": questions[r["qid"]],
                "response": response,
                "retrieved_contexts": [parents[p] for p in r["contexts"]],
                "reference": references[r["qid"]],
            }
        )
    return samples


def make_llm(base_url: str, model: str, api_key: str = "EMPTY", max_tokens: int = 1024):
    """Evaluator LLM for RAGAS behind an OpenAI-compatible endpoint. MD_JSON mode asks for JSON
    in a fenced block, which works without server-side structured-output support."""
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return llm_factory(
        model,
        provider="openai",
        client=client,
        mode=instructor.Mode.MD_JSON,
        temperature=0.0,
        max_tokens=max_tokens,
    )


def make_embeddings(model: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=model, device=device)


def make_metrics(llm, embeddings) -> dict:
    from ragas.metrics.collections import AnswerCorrectness, ContextRecall, Faithfulness

    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }


async def _score_one(sample: dict, metrics: dict) -> dict:
    out = {"qid": sample["qid"]}
    calls = {
        "faithfulness": lambda m: m.ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        ),
        "context_recall": lambda m: m.ascore(
            user_input=sample["user_input"],
            retrieved_contexts=sample["retrieved_contexts"],
            reference=sample["reference"],
        ),
        "answer_correctness": lambda m: m.ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            reference=sample["reference"],
        ),
    }
    for name, metric in metrics.items():
        try:
            out[name] = float((await calls[name](metric)).value)
        except Exception as e:  # noqa: BLE001  - a bad evaluator reply must not stop the run
            out[name] = float("nan")
            out[f"{name}_error"] = f"{type(e).__name__}: {e}"[:300]
    return out


async def score_samples(samples: Sequence[dict], metrics: dict, concurrency: int = 2) -> list[dict]:
    """Score samples with bounded concurrency; failures become NaN with the error recorded."""
    sem = asyncio.Semaphore(concurrency)

    async def run(sample):
        async with sem:
            return await _score_one(sample, metrics)

    return list(await asyncio.gather(*(run(s) for s in samples)))


class EvaluatorDown(RuntimeError):
    """The evaluator endpoint is unreachable or every metric failed on a whole batch."""


def check_endpoint(base_url: str, timeout: float = 10.0) -> None:
    """Fail fast on a wrong URL or a server that is down, before any sample is scored."""
    import httpx

    try:
        httpx.get(base_url.rstrip("/") + "/models", timeout=timeout).raise_for_status()
    except httpx.HTTPError as e:
        raise EvaluatorDown(f"evaluator at {base_url} not usable: {e!r}") from e


def all_failed(row: dict, metrics: Sequence[str]) -> bool:
    return all(math.isnan(row.get(m, float("nan"))) for m in metrics)


async def score_in_batches(
    samples: Sequence[dict], metrics: dict, batch: int, workers: int, on_batch
) -> None:
    """Score in batches inside ONE event loop (the async client is bound to it) and hand each
    batch's rows to `on_batch`. Rows where every metric failed are not handed on, so a resumed
    run retries them; a batch where every row failed means a systematic problem (wrong model,
    server down) and stops the run with the first error."""
    for i in range(0, len(samples), batch):
        rows = await score_samples(samples[i : i + batch], metrics, workers)
        ok = [r for r in rows if not all_failed(r, list(metrics))]
        if rows and not ok:
            first = next((v for k, v in rows[0].items() if k.endswith("_error")), "unknown")
            raise EvaluatorDown(f"every metric failed on a whole batch; first error: {first}")
        on_batch(ok)


def summarise_scores(scores: Sequence[dict], metrics: Sequence[str] = METRICS) -> dict:
    """Mean of each metric over the samples where it could be computed, and how many failed."""
    out: dict = {"n": len(scores)}
    for m in metrics:
        vals = [s[m] for s in scores if m in s and not math.isnan(s[m])]
        out[m] = sum(vals) / len(vals) if vals else float("nan")
        out[f"{m}_failed"] = len(scores) - len(vals)
    return out
