"""Batched cross-encoder reranking of retrieved chunks."""

from __future__ import annotations

from typing import Protocol

from .encoders import resolve_device


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder; all (query, passage) pairs scored in one batched pass."""

    def __init__(
        self, model_name: str, device: str = "auto", batch_size: int = 64, max_length: int = 512
    ):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.device = resolve_device(device)
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        # fp16 on GPU (scores match fp32 to ~1e-3 on this task); recorded in results metadata.
        self.dtype = "fp16" if self.device == "cuda" else "fp32"
        if self.dtype == "fp16":
            self.model.model.half()
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict(
            [(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
