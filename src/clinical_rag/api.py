"""HTTP API: `/search` (retrieval + rerank) and `/answer` (generation).

`create_app(...)` takes ready-made components, so tests run it with fakes. `uvicorn
clinical_rag.api:app` builds the real ones from settings on first use:
- Qdrant at `QDRANT_URL`, encoders and reranker from `CRAG_*` settings
- the generator at `CRAG_LLM_BASE_URL` / `CRAG_LLM_MODEL`; without them `/answer` returns 503.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .generate import LLMClient, LLMError
from .rag import RAGPipeline
from .retrieve import Retriever

DEFAULT_SYSTEM = "hybrid_dbsf+rerank_parent"
# Backend failures that are not the caller's fault: network/server errors from Qdrant or the
# generator, missing model or index files (OSError covers FileNotFoundError, ConnectionError).
BACKEND_ERRORS = (httpx.HTTPError, OSError, UnexpectedResponse, ResponseHandlingException)


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
    system: str = DEFAULT_SYSTEM
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
    """Build the app. Components are created once, on the first request that needs them.

    Errors: bad input or an unknown system -> 422; generator failure -> 502; retrieval backend
    (Qdrant, model files, index data) unavailable -> 503.
    """
    app = FastAPI(title="clinical-rag-eval", version="0.3.0")
    cache: dict = {}
    lock = threading.Lock()  # sync endpoints run in a thread pool

    def get(name, factory):
        if name not in cache:
            with lock:
                if name not in cache:  # another request may have built it meanwhile
                    try:
                        cache[name] = factory()
                    except BACKEND_ERRORS as e:
                        raise HTTPException(503, f"{name} unavailable: {e}") from e
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
        except BACKEND_ERRORS as e:
            raise HTTPException(503, f"retrieval backend unavailable: {e}") from e
        # Hits are best-first and scored by the last stage (the reranker when one ran);
        # keep the first score per abstract. Abstracts without text are skipped.
        best: dict[str, float] = {}
        for h in res.hits:
            best.setdefault(h.pmid, h.score)
        abstracts = [
            Abstract(pmid=p, score=best[p], text=parents[p])
            for p in res.ranked_pmids()
            if p in parents
        ][: req.k]
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
            system=req.system,
            top_k=req.top_k,
            rewrite=req.rewrite,
        )
        try:
            out = pipe.run(req.question)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except LLMError as e:
            raise HTTPException(502, f"generator error: {e}") from e
        except BACKEND_ERRORS as e:
            raise HTTPException(503, f"backend unavailable: {e}") from e
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
