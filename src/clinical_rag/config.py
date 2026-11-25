"""Central settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(_env("CRAG_DATA_DIR", str(REPO_ROOT / "data")))
    )
    results_dir: Path = field(
        default_factory=lambda: Path(_env("CRAG_RESULTS_DIR", str(REPO_ROOT / "results")))
    )
    # "http://localhost:6333" for the Docker server, ":memory:" or a folder path for embedded mode.
    qdrant_location: str = field(
        default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333")
    )
    collection: str = field(default_factory=lambda: _env("CRAG_COLLECTION", "pubmedqa"))
    dense_model: str = field(
        default_factory=lambda: _env("CRAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    )
    # bge v1.5 recommends this instruction for short queries against passages.
    dense_query_prefix: str = field(
        default_factory=lambda: _env(
            "CRAG_DENSE_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        )
    )
    sparse_model: str = field(default_factory=lambda: _env("CRAG_SPARSE_MODEL", "Qdrant/bm25"))
    device: str = field(default_factory=lambda: _env("CRAG_DEVICE", "auto"))
    max_chunk_words: int = field(default_factory=lambda: int(_env("CRAG_MAX_CHUNK_WORDS", "200")))


def get_settings() -> Settings:
    return Settings()
