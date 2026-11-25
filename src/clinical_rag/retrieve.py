"""Retrieval systems for the ablation: BM25 (sparse), dense, Qdrant hybrid (RRF or DBSF fusion).

Every search returns chunk hits plus per-stage timings in milliseconds, so latency can be reported
per stage (query encoding vs Qdrant search) and end to end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from .encoders import DenseEncoder, SparseEncoder
from .index import DENSE, SPARSE

SYSTEMS = ("bm25", "dense", "hybrid_rrf", "hybrid_dbsf")


@dataclass
class Hit:
    chunk_id: int
    pmid: str
    score: float
    text: str = ""


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)

    def ranked_pmids(self) -> list[str]:
        """Parent-level ranking: first (= best-scoring) occurrence of each PMID."""
        seen, out = set(), []
        for h in self.hits:
            if h.pmid not in seen:
                seen.add(h.pmid)
                out.append(h.pmid)
        return out


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        prefetch_limit: int = 100,
    ):
        self.client = client
        self.collection = collection
        self.dense = dense
        self.sparse = sparse
        self.prefetch_limit = prefetch_limit

    def _to_hits(self, points) -> list[Hit]:
        return [
            Hit(
                chunk_id=int(p.id),
                pmid=str(p.payload["pmid"]),
                score=float(p.score),
                text=p.payload.get("text", ""),
            )
            for p in points
        ]

    def search(
        self, system: str, query: str, limit: int = 50, with_text: bool = False
    ) -> SearchResult:
        """Run one retrieval system (see SYSTEMS) and return chunk hits with stage timings."""
        return self._search_base(system, query, limit=limit, with_text=with_text)

    def _search_base(self, system: str, query: str, limit: int, with_text: bool) -> SearchResult:
        t: dict[str, float] = {}
        payload = ["pmid", "text"] if with_text else ["pmid"]
        t_all = time.perf_counter()
        if system == "dense":
            t0 = time.perf_counter()
            qd = self.dense.encode_query(query)
            t["embed_dense"] = _ms(t0)
            t0 = time.perf_counter()
            res = self.client.query_points(
                self.collection, query=qd.tolist(), using=DENSE, limit=limit, with_payload=payload
            )
            t["search"] = _ms(t0)
        elif system == "bm25":
            t0 = time.perf_counter()
            qs = self.sparse.encode_query(query)
            t["embed_sparse"] = _ms(t0)
            t0 = time.perf_counter()
            res = self.client.query_points(
                self.collection,
                query=models.SparseVector(indices=qs.indices, values=qs.values),
                using=SPARSE,
                limit=limit,
                with_payload=payload,
            )
            t["search"] = _ms(t0)
        elif system in ("hybrid_rrf", "hybrid_dbsf"):
            t0 = time.perf_counter()
            qd = self.dense.encode_query(query)
            t["embed_dense"] = _ms(t0)
            t0 = time.perf_counter()
            qs = self.sparse.encode_query(query)
            t["embed_sparse"] = _ms(t0)
            fusion = models.Fusion.RRF if system == "hybrid_rrf" else models.Fusion.DBSF
            t0 = time.perf_counter()
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=qd.tolist(), using=DENSE, limit=self.prefetch_limit),
                    models.Prefetch(
                        query=models.SparseVector(indices=qs.indices, values=qs.values),
                        using=SPARSE,
                        limit=self.prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=fusion),
                limit=limit,
                with_payload=payload,
            )
            t["search"] = _ms(t0)
        else:
            raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
        t["total"] = _ms(t_all)
        return SearchResult(hits=self._to_hits(res.points), timings_ms=t)
