"""Hierarchical chunking: abstract (parent) -> labelled sections -> sentence windows (children).

PubMedQA contexts already come split into labelled sections (BACKGROUND, METHODS, RESULTS, ...).
Each section becomes one child chunk; a section longer than `max_words` is split into windows of
whole sentences. Retrieval runs over children and is scored per parent (PMID); generation gets
the full parent abstract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Document:
    pmid: str
    sections: list[tuple[str, str]]  # (label, text)
    split: str = ""

    @property
    def text(self) -> str:
        return "\n".join(f"{lab}: {txt}" if lab else txt for lab, txt in self.sections)


@dataclass
class Chunk:
    chunk_id: int
    pmid: str
    section: str
    position: int  # order of the chunk inside its parent
    text: str
    meta: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def window_sentences(sentences: list[str], max_words: int) -> list[str]:
    """Greedily pack whole sentences into windows of at most `max_words` words.

    A single sentence longer than `max_words` becomes its own window (never cut mid-sentence).
    """
    windows, cur, cur_len = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if cur and cur_len + n > max_words:
            windows.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += n
    if cur:
        windows.append(" ".join(cur))
    return windows


def chunk_document(doc: Document, max_words: int = 200, start_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    pos = 0
    for label, text in doc.sections:
        text = text.strip()
        if not text:
            continue
        pieces = (
            [text]
            if len(text.split()) <= max_words
            else window_sentences(split_sentences(text), max_words)
        )
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=start_id + len(chunks),
                    pmid=doc.pmid,
                    section=label,
                    position=pos,
                    text=piece,
                )
            )
            pos += 1
    return chunks


def chunk_corpus(docs: list[Document], max_words: int = 200) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_words=max_words, start_id=len(out)))
    return out
