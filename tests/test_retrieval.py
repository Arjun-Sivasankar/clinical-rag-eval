"""End-to-end retrieval over an in-memory Qdrant with deterministic fake encoders (no downloads)."""

import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from clinical_rag.chunking import Document, chunk_corpus
from clinical_rag.encoders import SparseVec
from clinical_rag.evaluate import evaluate_system, format_table
from clinical_rag.index import create_collection, make_client, upsert_chunks
from clinical_rag.retrieve import SYSTEMS, Retriever

DIM = 64


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _tid(tok):
    return int(hashlib.md5(tok.encode()).hexdigest()[:6], 16)


class FakeDense:
    """Bag-of-words hashed into DIM dims and normalised: similar wording -> similar vector."""

    name, dim, device = "fake-dense", DIM, "cpu"

    def _vec(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for t in _tokens(text):
            v[_tid(t) % DIM] += 1
        n = np.linalg.norm(v)
        return v / n if n else v + 1 / np.sqrt(DIM)

    def encode_docs(self, texts, batch_size=128):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


class FakeSparse:
    name = "fake-bm25"

    def _vec(self, text, query=False):
        counts = {}
        for t in _tokens(text):
            counts[_tid(t)] = counts.get(_tid(t), 0) + 1.0
        if query:
            counts = {k: 1.0 for k in counts}
        idx = sorted(counts)
        return SparseVec(indices=idx, values=[counts[i] for i in idx])

    def encode_docs(self, texts, batch_size=256):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text, query=True)


DOCS = [
    Document(
        "100",
        [
            ("BACKGROUND", "Statins lower cholesterol in adults."),
            ("RESULTS", "LDL fell by a third."),
        ],
    ),
    Document(
        "200",
        [
            ("BACKGROUND", "Aspirin and bleeding risk in the elderly."),
            ("RESULTS", "Bleeding rose."),
        ],
    ),
    Document(
        "300",
        [("BACKGROUND", "Vitamin D supplements and bone fractures."), ("RESULTS", "No effect.")],
    ),
    Document(
        "400",
        [("BACKGROUND", "Exercise improves depression scores."), ("RESULTS", "Scores improved.")],
    ),
]
QUERIES = pd.DataFrame(
    {
        "qid": ["100", "200", "300", "400"],
        "question": [
            "Do statins lower cholesterol?",
            "Does aspirin increase bleeding risk in the elderly?",
            "Do vitamin D supplements prevent fractures?",
            "Does exercise improve depression?",
        ],
        "gold_pmid": ["100", "200", "300", "400"],
    }
)


@pytest.fixture(scope="module")
def retriever():
    client = make_client(":memory:")
    chunks = chunk_corpus(DOCS)
    dense, sparse = FakeDense(), FakeSparse()
    texts = [c.text for c in chunks]
    create_collection(client, "test", dim=DIM)
    upsert_chunks(client, "test", chunks, dense.encode_docs(texts), sparse.encode_docs(texts))
    return Retriever(client, "test", dense, sparse, prefetch_limit=20)


@pytest.mark.parametrize("system", SYSTEMS)
def test_each_system_finds_gold(retriever, system):
    res = retriever.search(system, "Does aspirin increase bleeding risk?", limit=10)
    assert res.ranked_pmids()[0] == "200"
    assert {"search", "total"} <= set(res.timings_ms)


def test_ranked_pmids_dedupes_chunks(retriever):
    pmids = retriever.search("dense", "bleeding", limit=8).ranked_pmids()
    assert len(pmids) == len(set(pmids))


def test_evaluate_and_table(retriever):
    summary, rows = evaluate_system(retriever, "hybrid_rrf", QUERIES, chunk_limit=10, warmup=1)
    assert summary["recall@1"] == 1.0
    assert summary["ndcg@10"] == 1.0
    assert len(rows) == 4 and rows[0]["gold_rank"] == 1
    table = format_table([summary])
    assert "hybrid_rrf" in table and "1.000" in table


def test_unknown_system(retriever):
    with pytest.raises(ValueError):
        retriever.search("nope", "x")


@pytest.mark.slow
def test_bm25_doc_length_matches_fastembed_weights():
    """Our |d| must equal the one fastembed uses: with b=1 and avg_len=|d| a single-occurrence
    term gets weight exactly 1."""
    from clinical_rag.encoders import FastEmbedBM25Encoder

    text = "Statins lowered the LDL cholesterol of the older adults in this randomised trial."
    n = FastEmbedBM25Encoder().doc_lengths([text])[0]
    enc = FastEmbedBM25Encoder(avg_len=float(n))
    enc.model.model.b = 1.0
    vec = enc.encode_docs([text])[0]
    assert n > 0 and all(abs(v - 1.0) < 1e-6 for v in vec.values)


def test_evaluate_systems_interleaves_and_keeps_one_result_per_query(retriever):
    from clinical_rag.evaluate import evaluate_systems

    calls = []
    orig = retriever.search

    def spy(system, query, limit=50, with_text=False):
        calls.append(system)
        return orig(system, query, limit=limit, with_text=with_text)

    retriever.search = spy
    try:
        out = evaluate_systems(retriever, ["bm25", "dense"], QUERIES, chunk_limit=10, warmup=1)
    finally:
        del retriever.search
    timed = calls[2:]  # after one warm-up query per system
    n = len(QUERIES)
    expected = [
        s for i in range(n) for s in (["bm25", "dense"] if i % 2 == 0 else ["dense", "bm25"])
    ]
    assert timed == expected
    assert [s["system"] for s, _ in out] == ["bm25", "dense"]
    for (summary, rows), system in zip(out, ["bm25", "dense"], strict=True):
        # same scores and rows as evaluating that system on its own, in qid order
        alone_summary, alone_rows = evaluate_system(retriever, system, QUERIES, 10, warmup=0)
        assert [r["qid"] for r in rows] == QUERIES["qid"].tolist()
        assert rows == alone_rows
        for metric in ("recall@1", "recall@5", "mrr@10", "ndcg@10"):
            assert summary[metric] == alone_summary[metric]


def test_evaluate_systems_rejects_duplicates(retriever):
    from clinical_rag.evaluate import evaluate_systems

    with pytest.raises(ValueError):
        evaluate_systems(retriever, ["bm25", "bm25"], QUERIES)


class FakeReranker:
    """Scores by word overlap with the query, so it can reorder a noisy base ranking."""

    name = "fake-rerank"

    def score(self, query, passages):
        q = set(_tokens(query))
        return [len(q & set(_tokens(p))) for p in passages]


def test_rerank_stage(retriever):
    retriever.reranker, retriever.rerank_depth = FakeReranker(), 8
    try:
        res = retriever.search("hybrid_rrf+rerank", "vitamin d supplements fractures", limit=5)
        assert res.ranked_pmids()[0] == "300"
        assert "rerank" in res.timings_ms and len(res.hits) <= 5
        assert res.hits == sorted(res.hits, key=lambda h: h.score, reverse=True)
        summary, _ = evaluate_system(retriever, "dense+rerank", QUERIES, chunk_limit=10, warmup=0)
        assert "rerank" in summary["latency_ms"]
        assert "Rerank p50" in format_table([summary])
    finally:
        retriever.reranker = None


def test_rerank_without_model_fails(retriever):
    with pytest.raises(ValueError):
        retriever.search("dense+rerank", "x")


def test_rerank_parent_scores_whole_abstracts(retriever):
    parents = {d.pmid: d.text for d in DOCS}
    retriever.reranker, retriever.parents, retriever.rerank_parents = FakeReranker(), parents, 3
    try:
        res = retriever.search("hybrid_rrf+rerank_parent", "aspirin bleeding elderly", limit=10)
        pmids = [h.pmid for h in res.hits]
        assert pmids[0] == "200"
        assert len(pmids) == len(set(pmids))  # one hit per abstract
        assert set(pmids) == {"100", "200", "300", "400"}  # tail beyond the cut is kept
    finally:
        retriever.reranker, retriever.parents = None, None


def test_rerank_parent_needs_parents(retriever):
    retriever.reranker = FakeReranker()
    try:
        with pytest.raises(ValueError):
            retriever.search("dense+rerank_parent", "x")
    finally:
        retriever.reranker = None


def test_rerank_keeps_tail_in_base_order(retriever):
    retriever.reranker, retriever.rerank_depth = FakeReranker(), 2
    try:
        q = "aspirin bleeding risk elderly"
        base = [h.chunk_id for h in retriever.search("dense", q, limit=8).hits]
        res = retriever.search("dense+rerank", q, limit=8)
        ids = [h.chunk_id for h in res.hits]
        assert sorted(ids[:2]) == sorted(base[:2])  # head = reranked top 2
        assert ids[2:] == base[2:]  # tail untouched, in base order
    finally:
        retriever.reranker, retriever.rerank_depth = None, 50
