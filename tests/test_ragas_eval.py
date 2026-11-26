"""RAGAS wiring with a stub evaluator LLM and hand-checked verdicts.

The pure helpers run everywhere. The scoring tests need the `eval` extra (ragas) and are
skipped when it is not installed (as in CI).
"""

import asyncio
import math

import pytest

from clinical_rag.ragas_eval import samples_from_generations, summarise_scores

ROWS = [
    {
        "qid": "1",
        "raw": "<think>x</think>\nAnswer: yes\nExplanation: Statins lower LDL [10].",
        "contexts": ["10", "20"],
        "query": "statins ldl",
    },
    {"qid": "2", "raw": "", "contexts": ["20"], "query": "q"},  # empty reply: skipped
]
QUESTIONS = {"1": "Do statins lower LDL?", "2": "Q2?"}
REFERENCES = {"1": "Statins reduced LDL by a third.", "2": "ref"}
PARENTS = {"10": "Statins lowered LDL cholesterol.", "20": "Unrelated abstract."}


def test_samples_from_generations():
    s = samples_from_generations(ROWS, QUESTIONS, REFERENCES, PARENTS)
    assert len(s) == 1
    assert s[0] == {
        "qid": "1",
        "user_input": "Do statins lower LDL?",  # original question, not the rewritten query
        "response": "Answer: yes\nExplanation: Statins lower LDL [10].",  # think block removed
        "retrieved_contexts": ["Statins lowered LDL cholesterol.", "Unrelated abstract."],
        "reference": "Statins reduced LDL by a third.",
    }


def test_summarise_scores_ignores_failures():
    scores = [
        {"qid": "1", "faithfulness": 1.0, "context_recall": 0.5, "answer_correctness": 0.2},
        {"qid": "2", "faithfulness": 0.0, "context_recall": math.nan, "answer_correctness": 0.4},
    ]
    s = summarise_scores(scores)
    assert s["n"] == 2 and s["faithfulness"] == 0.5 and s["context_recall"] == 0.5
    assert s["context_recall_failed"] == 1 and s["answer_correctness"] == pytest.approx(0.3)


ragas = pytest.importorskip("ragas")


class StubLLM(ragas.llms.base.InstructorBaseRagasLLM):
    """Answers each RAGAS prompt type with a fixed, hand-chosen structured output."""

    def __init__(self, replies):
        self.replies = replies  # response-model class name -> dict
        self.calls = []

    def generate(self, prompt, response_model):
        self.calls.append(response_model.__name__)
        return response_model.model_validate(self.replies[response_model.__name__])

    async def agenerate(self, prompt, response_model):
        return self.generate(prompt, response_model)


REPLIES = {
    # faithfulness + answer correctness: answer split into 2 statements
    "StatementGeneratorOutput": {"statements": ["Statins lower LDL.", "They cure cancer."]},
    # faithfulness: 1 of 2 statements supported by the abstracts -> 0.5
    "NLIStatementOutput": {
        "statements": [
            {"statement": "Statins lower LDL.", "reason": "stated", "verdict": 1},
            {"statement": "They cure cancer.", "reason": "not stated", "verdict": 0},
        ]
    },
    # context recall: 2 of 3 reference statements attributable -> 2/3
    "ContextRecallOutput": {
        "classifications": [
            {"statement": "a", "reason": "r", "attributed": 1},
            {"statement": "b", "reason": "r", "attributed": 1},
            {"statement": "c", "reason": "r", "attributed": 0},
        ]
    },
    # answer correctness: TP=1, FP=1, FN=1 -> F1 = 1 / (1 + 0.5 * (1 + 1)) = 0.5
    "ClassificationWithReason": {
        "TP": [{"statement": "Statins lower LDL.", "reason": "r"}],
        "FP": [{"statement": "They cure cancer.", "reason": "r"}],
        "FN": [{"statement": "by a third", "reason": "r"}],
    },
}


def _metrics(llm):
    from ragas.metrics.collections import AnswerCorrectness, ContextRecall, Faithfulness

    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        # factuality only, so no embedding model is needed in the test
        "answer_correctness": AnswerCorrectness(llm=llm, weights=[1.0, 0.0]),
    }


def test_scores_match_hand_computed_values():
    from clinical_rag.ragas_eval import score_samples

    llm = StubLLM(REPLIES)
    samples = samples_from_generations(ROWS, QUESTIONS, REFERENCES, PARENTS)
    [s] = asyncio.run(score_samples(samples, _metrics(llm)))
    assert s["qid"] == "1"
    assert s["faithfulness"] == pytest.approx(0.5)
    assert s["context_recall"] == pytest.approx(2 / 3)
    assert s["answer_correctness"] == pytest.approx(0.5)
    assert "NLIStatementOutput" in llm.calls and "ContextRecallOutput" in llm.calls


def test_bad_evaluator_reply_becomes_nan_with_error():
    from clinical_rag.ragas_eval import score_samples

    broken = dict(REPLIES, NLIStatementOutput={"statements": "not a list"})
    samples = samples_from_generations(ROWS, QUESTIONS, REFERENCES, PARENTS)
    [s] = asyncio.run(score_samples(samples, _metrics(StubLLM(broken))))
    assert math.isnan(s["faithfulness"]) and "faithfulness_error" in s
    assert s["context_recall"] == pytest.approx(2 / 3)  # other metrics still scored


def test_whole_batch_failure_stops_and_saves_nothing():
    from clinical_rag.ragas_eval import EvaluatorDown, score_in_batches

    class DownLLM(StubLLM):
        def generate(self, prompt, response_model):
            raise ConnectionError("evaluator down")

    samples = samples_from_generations(ROWS, QUESTIONS, REFERENCES, PARENTS)
    saved = []
    with pytest.raises(EvaluatorDown, match="evaluator down"):
        asyncio.run(score_in_batches(samples, _metrics(DownLLM({})), 1, 1, saved.extend))
    assert saved == []  # nothing marked done, so a resumed run retries


def test_batches_save_partial_failures_and_skip_all_failed_rows():
    from clinical_rag.ragas_eval import all_failed, score_in_batches

    samples = samples_from_generations(ROWS, QUESTIONS, REFERENCES, PARENTS) * 2
    saved = []
    asyncio.run(score_in_batches(samples, _metrics(StubLLM(REPLIES)), 1, 1, saved.extend))
    names = ["faithfulness", "context_recall", "answer_correctness"]
    assert len(saved) == 2 and not any(all_failed(r, names) for r in saved)


def test_missing_parent_abstract_is_a_clear_error():
    with pytest.raises(KeyError, match="not in docs.parquet"):
        samples_from_generations(ROWS, QUESTIONS, REFERENCES, {"10": "only one"})


def test_check_endpoint_fails_fast():
    from clinical_rag.ragas_eval import EvaluatorDown, check_endpoint

    with pytest.raises(EvaluatorDown):
        check_endpoint("http://127.0.0.1:9/v1", timeout=0.5)  # nothing listens on port 9
