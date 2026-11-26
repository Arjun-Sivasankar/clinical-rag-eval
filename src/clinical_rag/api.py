"""HTTP API: `/search` (retrieval + rerank) and `/answer` (generation).

`create_app(...)` takes ready-made components, so tests run it with fakes. `uvicorn
clinical_rag.api:app` builds the real ones from settings on first use:
- Qdrant at `QDRANT_URL`, encoders and reranker from `CRAG_*` settings
- the generator at `CRAG_LLM_BASE_URL` / `CRAG_LLM_MODEL`; without them `/answer` returns 503.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import LLMClient
from .rag import RAGPipeline
from .retrieve import Retriever

DEFAULT_SYSTEM = "hybrid_dbsf+rerank_parent"


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    system: str = DEFAULT_SYSTEM
    k: int = Field(5, ge=1, le=20)


class Abstract(BaseModel):
    pmid: str
    score: float
    text: str


class SearchResponse(BaseModel):
    system: str
    abstracts: list[Abstract]
    timings_ms: dict[str, float]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    rewrite: bool = False
    top_k: int = Field(3, ge=1, le=10)


class AnswerResponse(BaseModel):
    label: Literal["yes", "no", "maybe"] | None
    explanation: str
    query: str
    contexts: list[str]
    cited: list[str]
    timings_ms: dict[str, float]


def create_app(
    retriever_factory: Callable[[], Retriever],
    parents_factory: Callable[[], Mapping[str, str]],
    llm_factory: Callable[[], LLMClient | None] = lambda: None,
) -> FastAPI:
    """Build the app. Components are created once, on the first request that needs them."""
    app = FastAPI(title="clinical-rag-eval", version="0.3.0")
    cache: dict = {}

    def get(name, factory):
        if name not in cache:
            cache[name] = factory()
        return cache[name]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        retriever = get("retriever", retriever_factory)
        parents = get("parents", parents_factory)
        try:
            res = retriever.search(req.system, req.question, limit=100)
        except ValueError as e:  # unknown system or missing reranker
            raise HTTPException(status_code=422, detail=str(e)) from e
        best: dict[str, float] = {}
        for h in res.hits:  # hits are best-first; keep the first (best) score per abstract
            best.setdefault(h.pmid, h.score)
        abstracts = [
            Abstract(pmid=p, score=best[p], text=parents.get(p, ""))
            for p in res.ranked_pmids()[: req.k]
        ]
        return SearchResponse(system=req.system, abstracts=abstracts, timings_ms=res.timings_ms)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        llm = get("llm", llm_factory)
        if llm is None:
            raise HTTPException(
                status_code=503, detail="no generator configured (set CRAG_LLM_BASE_URL)"
            )
        pipe = RAGPipeline(
            get("retriever", retriever_factory),
            llm,
            get("parents", parents_factory),
            system=DEFAULT_SYSTEM,
            top_k=req.top_k,
            rewrite=req.rewrite,
        )
        out = pipe.run(req.question)
        return AnswerResponse(
            label=out["label"],
            explanation=out["explanation"],
            query=out["query"],
            contexts=out["contexts"],
            cited=out["cited"],
            timings_ms=out["timings_ms"],
        )

    return app


def _real_retriever() -> Retriever:
    import json

    from .config import get_settings
    from .encoders import FastEmbedBM25Encoder, SentenceTransformerEncoder
    from .index import make_client
    from .rerank import CrossEncoderReranker

    s = get_settings()
    meta_path = s.data_dir / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return Retriever(
        make_client(s.qdrant_location),
        s.collection,
        SentenceTransformerEncoder(s.dense_model, s.dense_query_prefix, s.device),
        FastEmbedBM25Encoder(s.sparse_model, avg_len=meta.get("bm25_avg_len")),
        reranker=CrossEncoderReranker(
            os.environ.get("CRAG_RERANKER", "ncbi/MedCPT-Cross-Encoder"), s.device
        ),
        parents=_real_parents(),
    )


@lru_cache(maxsize=1)
def _real_parents() -> Mapping[str, str]:
    import pandas as pd

    from .config import get_settings

    df = pd.read_parquet(get_settings().data_dir / "docs.parquet", columns=["pmid", "text"])
    return dict(zip(df["pmid"], df["text"], strict=True))


def _real_llm() -> LLMClient | None:
    base_url = os.environ.get("CRAG_LLM_BASE_URL")
    if not base_url:
        return None
    return LLMClient(base_url, os.environ.get("CRAG_LLM_MODEL", "Qwen/Qwen3-1.7B"))


app = create_app(_real_retriever, _real_parents, _real_llm)
