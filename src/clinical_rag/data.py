"""PubMedQA download and corpus construction.

PubMedQA (MIT licence): https://github.com/pubmedqa/pubmedqa, mirrored by the authors on the
Hugging Face Hub as `qiaojin/PubMedQA`.

- queries: the 1,000 expert-labelled questions (`pqa_labeled`), each with one gold abstract (PMID),
  a yes/no/maybe label and a long answer (the abstract's conclusion).
- corpus: the abstract contexts of `pqa_labeled` + `pqa_unlabeled`, pooled, de-duplicated by PMID.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .chunking import Document

log = logging.getLogger(__name__)

HF_REPO = "qiaojin/PubMedQA"
SPLITS = ("pqa_labeled", "pqa_unlabeled")


def download(splits: tuple[str, ...] = SPLITS) -> dict[str, Path]:
    """Download the parquet files of the given splits into the Hugging Face cache (HF_HOME)."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"{s}/*.parquet" for s in splits],
        )
    )
    out = {}
    for split in splits:
        files = sorted((root / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files for {split} under {root}")
        out[split] = files[0] if len(files) == 1 else root / split
    return out


def read_split(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return pd.concat(
            [pd.read_parquet(p) for p in sorted(path.glob("*.parquet"))], ignore_index=True
        )
    return pd.read_parquet(path)


def _s(x) -> str:
    return "" if x is None else str(x)


def to_documents(df: pd.DataFrame, split: str) -> list[Document]:
    docs = []
    for row in df.itertuples(index=False):
        ctx = row.context
        texts = list(ctx["contexts"])
        labels = list(ctx["labels"]) if ctx.get("labels") is not None else [""] * len(texts)
        if len(labels) != len(texts):
            labels = [""] * len(texts)
        docs.append(
            Document(
                pmid=str(row.pubid),
                sections=[(_s(lab), _s(txt)) for lab, txt in zip(labels, texts, strict=True)],
                split=split,
            )
        )
    return docs


def to_queries(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qid": df["pubid"].astype(str),
            "question": df["question"].astype(str),
            "gold_pmid": df["pubid"].astype(str),
            "final_decision": df["final_decision"].astype(str),
            "long_answer": df["long_answer"].astype(str),
        }
    )


def build_corpus(split_paths: dict[str, Path]) -> tuple[list[Document], pd.DataFrame]:
    """Return (documents de-duplicated by PMID, queries from pqa_labeled).

    `pqa_labeled` is read first whatever the dict order, so a PMID present in both splits keeps
    its labelled copy (the one the gold answers refer to).
    """
    seen: set[str] = set()
    docs: list[Document] = []
    queries = None
    order = sorted(split_paths, key=lambda sp: sp != "pqa_labeled")
    for split in order:
        path = split_paths[split]
        df = read_split(path)
        log.info("%s: %d rows", split, len(df))
        if split == "pqa_labeled":
            queries = to_queries(df)
        for doc in to_documents(df, split):
            if doc.pmid in seen:
                continue
            seen.add(doc.pmid)
            docs.append(doc)
    if queries is None:
        raise ValueError("pqa_labeled is required for the queries")
    return docs, queries
