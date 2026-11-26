"""API tests with FastAPI's TestClient, the in-memory retriever and a fake LLM server."""

import httpx
import pytest
from fastapi.testclient import TestClient
from test_retrieval import DOCS, FakeDense, FakeReranker, FakeSparse

from clinical_rag.api import create_app
from clinical_rag.chunking import chunk_corpus
from clinical_rag.generate import LLMClient
from clinical_rag.index import create_collection, make_client, upsert_chunks
from clinical_rag.retrieve import Retriever

PARENTS = {d.pmid: d.text for d in DOCS}


@pytest.fixture(scope="module")
def retriever():
    client = make_client(":memory:")
    chunks = chunk_corpus(DOCS)
    dense, sparse = FakeDense(), FakeSparse()
    texts = [c.text for c in chunks]
    create_collection(client, "api", dim=dense.dim)
    upsert_chunks(client, "api", chunks, dense.encode_docs(texts), sparse.encode_docs(texts))
    return Retriever(
        client, "api", dense, sparse, prefetch_limit=20, reranker=FakeReranker(), parents=PARENTS
    )


def _llm(reply="Answer: yes\nExplanation: see [200]."):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return LLMClient("http://fake/v1", "m", transport=httpx.MockTransport(handler))


def _client(retriever, llm=None):
    return TestClient(create_app(lambda: retriever, lambda: PARENTS, lambda: llm))


def test_health(retriever):
    assert _client(retriever).get("/health").json() == {"status": "ok"}


def test_search_returns_abstracts(retriever):
    r = _client(retriever).post("/search", json={"question": "aspirin bleeding elderly", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["system"] == "hybrid_dbsf+rerank_parent"
    assert [a["pmid"] for a in body["abstracts"]][0] == "200"
    assert len(body["abstracts"]) == 2 and body["abstracts"][0]["text"] == PARENTS["200"]
    assert "rerank" in body["timings_ms"]


def test_search_validation_and_unknown_system(retriever):
    c = _client(retriever)
    assert c.post("/search", json={"question": "x"}).status_code == 422  # too short
    assert c.post("/search", json={"question": "aspirin", "k": 99}).status_code == 422
    r = c.post("/search", json={"question": "aspirin", "system": "nope"})
    assert r.status_code == 422 and "unknown system" in r.json()["detail"]


def test_answer(retriever):
    r = _client(retriever, _llm()).post("/answer", json={"question": "Is aspirin risky?"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "yes" and body["cited"] == ["200"]
    assert body["contexts"] and "generate" in body["timings_ms"]


def test_answer_without_generator_is_503(retriever):
    r = _client(retriever, None).post("/answer", json={"question": "Is aspirin risky?"})
    assert r.status_code == 503


def test_components_built_once(retriever):
    calls = []

    def factory():
        calls.append(1)
        return retriever

    c = TestClient(create_app(factory, lambda: PARENTS))
    c.post("/search", json={"question": "aspirin"})
    c.post("/search", json={"question": "statins"})
    assert len(calls) == 1
