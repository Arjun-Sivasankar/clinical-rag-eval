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


def test_concurrent_first_requests_build_components_once(retriever):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    calls, lock = [], threading.Lock()

    def slow_factory():
        with lock:
            calls.append(1)
        time.sleep(0.2)  # widen the race window
        return retriever

    c = TestClient(create_app(slow_factory, lambda: PARENTS))
    with ThreadPoolExecutor(4) as ex:
        codes = list(
            ex.map(
                lambda q: c.post("/search", json={"question": q}).status_code,
                ["aspirin", "statins", "vitamin d", "exercise"],
            )
        )
    assert codes == [200] * 4 and len(calls) == 1


def test_generator_down_is_502(retriever):
    def handler(request):
        raise httpx.ConnectError("refused")

    llm = LLMClient(
        "http://fake/v1", "m", retries=1, backoff_s=0, transport=httpx.MockTransport(handler)
    )
    r = _client(retriever, llm).post("/answer", json={"question": "Is aspirin risky?"})
    assert r.status_code == 502 and "unreachable" in r.json()["detail"]


def test_retrieval_backend_down_is_503(retriever):
    def broken_factory():
        raise FileNotFoundError("data/docs.parquet")

    c = TestClient(create_app(lambda: retriever, broken_factory))
    r = c.post("/search", json={"question": "aspirin"})
    assert r.status_code == 503 and "parents unavailable" in r.json()["detail"]


def test_answer_unknown_system_is_422(retriever):
    r = _client(retriever, _llm()).post(
        "/answer", json={"question": "Is aspirin risky?", "system": "nope"}
    )
    assert r.status_code == 422
