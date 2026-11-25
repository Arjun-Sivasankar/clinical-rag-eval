"""Generation and the RAG pipeline against a fake OpenAI-compatible server (no model, no GPU)."""

import json

import httpx
import pytest
from test_retrieval import DOCS, FakeDense, FakeSparse  # noqa: F401  (shared fakes)

from clinical_rag.chunking import chunk_corpus
from clinical_rag.generate import (
    LLMClient,
    answer_messages,
    cited_pmids,
    parse_explanation,
    parse_label,
    parse_rewrite,
    strip_think,
)
from clinical_rag.index import create_collection, make_client, upsert_chunks
from clinical_rag.rag import RAGPipeline, macro_f1, summarise
from clinical_rag.retrieve import Retriever


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Answer: yes\nExplanation: shown in [100].", "yes"),
        ("Answer: **No**\nExplanation: x", "no"),
        ("<think>maybe yes, maybe no</think>\nAnswer: maybe\nExplanation: unclear", "maybe"),
        ("Yes. The trial [100] shows it.", "yes"),
        ("The evidence is mixed.", None),
        ("<think>unterminated reasoning, answer: yes", None),
    ],
)
def test_parse_label(text, label):
    assert parse_label(text) == label


def test_strip_think_and_explanation():
    text = "<think>step 1\nstep 2</think>\nAnswer: no\nExplanation: Not effective [123456]."
    assert strip_think(text).startswith("Answer: no")
    assert parse_explanation(text) == "Not effective [123456]."
    assert cited_pmids(text) == ["123456"]


def test_parse_rewrite():
    assert parse_rewrite('Query: "statins LDL cholesterol"', "q") == "statins LDL cholesterol"
    assert parse_rewrite("<think>hmm</think>\n\naspirin bleeding elderly\nextra", "q") == (
        "aspirin bleeding elderly"
    )
    assert parse_rewrite("   ", "original question") == "original question"


def test_answer_messages_contain_contexts_and_no_think():
    msgs = answer_messages("Q?", [("111", "text one"), ("222", "text two")])
    user = msgs[-1]["content"]
    assert "[111]\ntext one" in user and "[222]\ntext two" in user
    assert user.endswith("/no_think")


def test_macro_f1():
    gold = ["yes", "no", "maybe", "yes"]
    assert macro_f1(gold, gold) == 1.0
    # yes: tp=1 fp=1 fn=1 -> 2/4; no: tp=1 -> 1; maybe: tp=0 fn=1 -> 0
    assert macro_f1(gold, ["yes", "no", "yes", None]) == pytest.approx((0.5 + 1 + 0) / 3)


def _fake_llm(replies):
    """LLMClient whose server returns the given replies in turn and records requests."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = replies[(len(seen) - 1) % len(replies)]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = LLMClient("http://fake/v1", "m", transport=httpx.MockTransport(handler))
    return client, seen


def test_llm_client_posts_openai_format():
    llm, seen = _fake_llm(["hello"])
    c = llm.chat([{"role": "user", "content": "hi"}], max_tokens=7)
    assert c.text == "hello" and c.completion_tokens == 5 and c.latency_ms >= 0
    assert seen[0]["model"] == "m" and seen[0]["max_tokens"] == 7 and seen[0]["temperature"] == 0


@pytest.fixture(scope="module")
def retriever():
    client = make_client(":memory:")
    chunks = chunk_corpus(DOCS)
    dense, sparse = FakeDense(), FakeSparse()
    texts = [c.text for c in chunks]
    create_collection(client, "gen", dim=dense.dim)
    upsert_chunks(client, "gen", chunks, dense.encode_docs(texts), sparse.encode_docs(texts))
    return Retriever(client, "gen", dense, sparse, prefetch_limit=20)


def test_pipeline_with_rewrite(retriever):
    llm, seen = _fake_llm(["aspirin bleeding elderly", "Answer: yes\nExplanation: see [200]."])
    parents = {d.pmid: d.text for d in DOCS}
    pipe = RAGPipeline(retriever, llm, parents, system="hybrid_rrf", top_k=2, rewrite=True)
    out = pipe.run("Is aspirin risky for older people?")
    assert out["query"] == "aspirin bleeding elderly"
    assert out["contexts"][0] == "200" and len(out["contexts"]) == 2
    assert out["label"] == "yes" and out["cited"] == ["200"]
    assert {"rewrite", "generate", "total", "retrieve_search"} <= set(out["timings_ms"])
    assert "[200]" in seen[1]["messages"][-1]["content"]  # gold abstract given as context


def test_summarise(retriever):
    llm, _ = _fake_llm(["Answer: yes\nExplanation: [200]"])
    parents = {d.pmid: d.text for d in DOCS}
    pipe = RAGPipeline(retriever, llm, parents, system="hybrid_rrf", top_k=1)
    rows = []
    for pmid, q, gold in [("200", "aspirin bleeding elderly", "yes"), ("300", "vitamin d", "no")]:
        r = pipe.run(q)
        rows.append({**r, "gold_pmid": pmid, "gold_label": gold})
    s = summarise(rows)
    assert s["n"] == 2 and s["accuracy"] == 0.5 and s["unparsed"] == 0
    assert s["pred_counts"] == {"yes": 2, "no": 0, "maybe": 0}
    assert s["recall@1"] == 1.0 and s["gold_in_context"] == 1.0
    assert "generate" in s["latency_ms"]
