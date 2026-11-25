"""Qdrant collection: one named dense vector and one named sparse (BM25, IDF) vector per chunk."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from .chunking import Chunk
from .encoders import SparseVec

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"


def make_client(location: str) -> QdrantClient:
    if location == ":memory:":
        return QdrantClient(location=":memory:")
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, timeout=120)
    return QdrantClient(path=location)


def create_collection(client: QdrantClient, name: str, dim: int, recreate: bool = True) -> None:
    if client.collection_exists(name):
        if not recreate:
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(
        name, field_name="pmid", field_schema=models.PayloadSchemaType.KEYWORD
    )


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    dense: np.ndarray,
    sparse: Sequence[SparseVec],
    batch_size: int = 512,
) -> None:
    assert len(chunks) == len(dense) == len(sparse), "chunks and vectors must align"
    for start in tqdm(range(0, len(chunks), batch_size), desc="upsert", disable=len(chunks) < 5000):
        end = start + batch_size
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={
                    DENSE: d.tolist(),
                    SPARSE: models.SparseVector(indices=s.indices, values=s.values),
                },
                payload={
                    "pmid": c.pmid,
                    "section": c.section,
                    "position": c.position,
                    "text": c.text,
                },
            )
            for c, d, s in zip(chunks[start:end], dense[start:end], sparse[start:end], strict=True)
        ]
        client.upsert(collection_name=name, points=points, wait=True)
