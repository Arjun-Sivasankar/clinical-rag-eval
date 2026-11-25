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
from clinical_rag.rag import RAGPipeline, macro_f1, open_run, summarise
from clinical_rag.retrieve import Retriever


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Answer: yes\nExplanation: shown in [100].", "yes"),
        ("Answer: **No**\nExplanation: x", "no"),
        ("<think>maybe yes, maybe no</think>\nAnswer: maybe\nExplanation: unclear", "maybe"),
        ("Yes. The trial [100] shows it.", "yes"),
        ("The evidence is mixed.", None),
        ("The answer is no, because [100] found no effect.", "no"),
        ("**Answer**: yes\n**Explanation**: [100]", "yes"),
        ("Answer: yes | no | maybe\nExplanation: unsure", None),  # echoed template
        ("Answer: Yes, but only in adults.", "yes"),
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
    reqs = []

    def handler(request):
        reqs.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    with LLMClient(
        "http://fake/v1/", "m", api_key="k", transport=httpx.MockTransport(handler)
    ) as llm:
        c = llm.chat([{"role": "user", "content": "hi"}], max_tokens=7)
    body = json.loads(reqs[0].content)
    assert c.text == "hello" and c.completion_tokens is None and c.latency_ms >= 0
    assert str(reqs[0].url) == "http://fake/v1/chat/completions"
    assert reqs[0].headers["authorization"] == "Bearer k"
    assert body["model"] == "m" and body["max_tokens"] == 7 and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_llm_client_retries_5xx_then_raises_with_message():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="loading")
        if len(calls) == 2:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(400, text="bad prompt")

    llm = LLMClient("http://fake/v1", "m", backoff_s=0, transport=httpx.MockTransport(handler))
    assert llm.chat([]).text == "ok" and len(calls) == 2  # 503 retried
    with pytest.raises(RuntimeError, match="400: bad prompt"):  # 4xx not retried
        llm.chat([])
    assert len(calls) == 3


def test_llm_client_gives_up_after_retries():
    llm = LLMClient(
        "http://fake/v1",
        "m",
        retries=2,
        backoff_s=0,
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="down")),
    )
    with pytest.raises(RuntimeError, match="500: down"):
        llm.chat([])


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


def test_pipeline_context_comes_from_reranked_ranking(retriever):
    from test_retrieval import FakeReranker

    llm, seen = _fake_llm(["Answer: no\nExplanation: [300]"])
    parents = {d.pmid: d.text for d in DOCS}
    retriever.reranker, retriever.parents, retriever.rerank_parents = FakeReranker(), parents, 4
    try:
        pipe = RAGPipeline(retriever, llm, parents, system="dense+rerank_parent", top_k=1)
        out = pipe.run("vitamin d supplements bone fractures")
        base = retriever.search("dense", "vitamin d supplements bone fractures").ranked_pmids()
    finally:
        retriever.reranker, retriever.parents = None, None
    assert out["contexts"] == ["300"]  # the reranker's top abstract
    assert out["ranked"][0] == "300" and "retrieve_rerank" in out["timings_ms"]
    assert set(out["ranked"]) == set(base)  # same candidates, reranked order
    assert parents["300"] in seen[0]["messages"][-1]["content"]


def test_open_run_resumes_and_refuses_other_settings(tmp_path):
    rows = tmp_path / "run.jsonl"
    cfg = {"model": "m", "top_k": 3}
    assert open_run(rows, cfg) == {}
    rows.write_text('{"qid": "1", "label": "yes"}\n{"qid": "2", "lab', encoding="utf-8")
    done = open_run(rows, cfg)  # partial last line dropped
    assert list(done) == ["1"]
    assert rows.read_text(encoding="utf-8") == '{"qid": "1", "label": "yes"}\n'
    with pytest.raises(SystemExit, match="top_k"):
        open_run(rows, {"model": "m", "top_k": 5})


def test_summarise_handles_rows_with_different_stages():
    base = {
        "gold_pmid": "1",
        "gold_label": "yes",
        "label": "yes",
        "raw": "Answer: yes",
        "contexts": ["1"],
        "cited": [],
        "ranked": ["1"],
    }
    rows = [
        {**base, "timings_ms": {"generate": 10.0}},
        {**base, "raw": "", "label": None, "timings_ms": {"generate": 20.0, "rewrite": 5.0}},
    ]
    s = summarise(rows)
    assert s["empty_replies"] == 1 and s["unparsed"] == 1
    assert s["latency_ms"]["rewrite"]["p50"] == 5.0
