import json

import httpx
import pytest

from clinical_rag.generate import LLMClient
from clinical_rag.judge import agreement, cohen_kappa, confusion, judge, judge_messages


def test_cohen_kappa_hand_computed():
    gold = ["yes", "yes", "no", "no"]
    assert cohen_kappa(gold, gold) == pytest.approx(1.0)
    # po = 0.5; gold yes/no 0.5/0.5, pred yes/no 0.5/0.5 -> pe = 0.5 -> kappa 0
    assert cohen_kappa(gold, ["yes", "no", "yes", "no"]) == pytest.approx(0.0)
    # po = 0.75; pe = 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa 0.5
    assert cohen_kappa(gold, ["yes", "yes", "yes", "no"]) == pytest.approx(0.5)


def test_unparsed_counts_as_disagreement():
    gold = ["yes", "no"]
    a = agreement(gold, ["yes", None])
    assert a["accuracy"] == 0.5 and a["unparsed"] == 1
    assert a["confusion"]["no"]["none"] == 1 and a["confusion"]["yes"]["yes"] == 1


def test_confusion_shape():
    m = confusion(["maybe"], ["no"])
    assert set(m) == {"yes", "no", "maybe"} and m["maybe"]["no"] == 1
    assert sum(sum(r.values()) for r in m.values()) == 1


def test_judge_reads_label_from_fake_server():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: maybe"}}]})

    llm = LLMClient("http://fake/v1", "j", transport=httpx.MockTransport(handler))
    label, raw, ms = judge(llm, "Does X work?", "Results were inconsistent across trials.")
    assert label == "maybe" and raw == "Answer: maybe" and ms >= 0
    user = seen[0]["messages"][-1]["content"]
    assert "Does X work?" in user and "inconsistent" in user and seen[0]["max_tokens"] == 16


def test_judge_messages_format():
    msgs = judge_messages("Q?", "T.")
    assert msgs[0]["role"] == "system" and "yes | no | maybe" in msgs[0]["content"]
    assert msgs[1]["content"].endswith("/no_think")
