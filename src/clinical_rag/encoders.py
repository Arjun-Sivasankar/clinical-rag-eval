"""Dense and sparse text encoders behind two small protocols, so tests can swap in fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class DenseEncoder(Protocol):
    dim: int
    name: str

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SparseEncoder(Protocol):
    name: str

    def encode_docs(self, texts: list[str], batch_size: int = 256) -> list[SparseVec]: ...

    def encode_query(self, text: str) -> SparseVec: ...


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


class SentenceTransformerEncoder:
    """Dense encoder (sentence-transformers). Normalised vectors, fp16 on GPU."""

    def __init__(self, model_name: str, query_prefix: str = "", device: str = "auto"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()
        self.query_prefix = query_prefix
        # Renamed in newer sentence-transformers; keep both for older installs.
        get_dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            self.query_prefix + text, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


class FastEmbedBM25Encoder:
    """BM25 term weights from fastembed (`Qdrant/bm25`); IDF is applied server-side by Qdrant.

    `avg_len` is BM25's average document length (in stemmed, stopword-filtered tokens) used for
    length normalisation. fastembed defaults to 256; pass the corpus mean (see `mean_doc_len`).
    It only affects document vectors: query terms all get weight 1.
    """

    def __init__(self, model_name: str = "Qdrant/bm25", avg_len: float | None = None):
        from fastembed import SparseTextEmbedding

        self.name = model_name
        kwargs = {} if avg_len is None else {"avg_len": avg_len}
        self.model = SparseTextEmbedding(model_name=model_name, **kwargs)
        self.avg_len = self.model.model.avg_len

    def doc_lengths(self, texts: list[str]) -> list[int]:
        """BM25 length of each text: tokens after fastembed's tokenizer, stopword removal and
        stemming, i.e. the |d| in its length normalisation. Uses fastembed internals, checked by
        a test against fastembed's own term weights."""
        inner = self.model.model
        return [len(inner._stem(inner.tokenizer.tokenize(t))) for t in texts]

    def mean_doc_len(self, texts: list[str]) -> float:
        lengths = self.doc_lengths(texts)
        return sum(lengths) / len(lengths)

    @staticmethod
    def _conv(e) -> SparseVec:
        return SparseVec(indices=e.indices.tolist(), values=e.values.tolist())

    def encode_docs(self, texts: list[str], batch_size: int = 256) -> list[SparseVec]:
        return [self._conv(e) for e in self.model.embed(texts, batch_size=batch_size)]

    def encode_query(self, text: str) -> SparseVec:
        # query_embed gives each query term weight 1 (BM25 query side).
        return self._conv(next(iter(self.model.query_embed(text))))
