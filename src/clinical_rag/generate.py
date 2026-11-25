"""Answer generation and query rewriting against any OpenAI-compatible chat endpoint.

The same code talks to vLLM (`vllm serve`, laptop GPU or Kaggle) and MLX-LM (`mlx_lm.server`,
Mac mini): both expose `POST {base_url}/chat/completions`. Qwen3 models think before answering by
default; prompts end with `/no_think` and any `<think>...</think>` block is stripped before
parsing, so the output format is the same whichever server runs the model.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

LABELS = ("yes", "no", "maybe")

ANSWER_SYSTEM = (
    "You answer biomedical research questions using only the abstracts provided. "
    "Reply in exactly this format:\n"
    "Answer: yes | no | maybe\n"
    "Explanation: one to three sentences, citing the supporting abstracts as [PMID].\n"
    "Say maybe when the abstracts are inconclusive or do not answer the question."
)

REWRITE_SYSTEM = (
    "Rewrite the user's biomedical question as a short keyword search query for PubMed: key "
    "medical terms, synonyms and abbreviations, no question words, at most 20 words. Reply with "
    "the query only."
)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_ANSWER = re.compile(r"answer\s*[:\-]\s*\**\s*(yes|no|maybe)\b", re.I)
_FIRST_WORD = re.compile(r"^\W*(yes|no|maybe)\b", re.I)


@dataclass
class Completion:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class LLMClient:
    """Minimal client for an OpenAI-compatible `/chat/completions` endpoint."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 120.0
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout,
            transport=self.transport,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def chat(
        self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0
    ) -> Completion:
        t0 = time.perf_counter()
        r = self._http.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def strip_think(text: str) -> str:
    """Remove Qwen3-style reasoning blocks, including an unterminated one at the end."""
    text = _THINK.sub("", text)
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


def answer_messages(question: str, contexts: Sequence[tuple[str, str]]) -> list[dict]:
    """Chat messages for a yes/no/maybe answer from (pmid, abstract text) contexts."""
    blocks = "\n\n".join(f"[{pmid}]\n{text}" for pmid, text in contexts)
    user = f"Abstracts:\n\n{blocks}\n\nQuestion: {question}\n/no_think"
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def rewrite_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"{question}\n/no_think"},
    ]


def parse_label(text: str) -> str | None:
    """yes/no/maybe from a model reply: the `Answer:` line, else a leading label; else None."""
    text = strip_think(text)
    m = _ANSWER.search(text) or _FIRST_WORD.search(text)
    return m.group(1).lower() if m else None


def parse_explanation(text: str) -> str:
    text = strip_think(text)
    m = re.search(r"explanation\s*[:\-]\s*(.*)", text, re.I | re.S)
    return (m.group(1) if m else text).strip()


def parse_rewrite(text: str, fallback: str) -> str:
    """First non-empty line of the reply, unquoted; the original question if empty."""
    for line in strip_think(text).splitlines():
        line = line.strip()
        if line.lower().startswith("query:"):
            line = line[6:].strip()
        line = line.strip('"').strip("'").strip()
        if line:
            return line
    return fallback


def cited_pmids(text: str) -> list[str]:
    """PMIDs cited as [12345678]; PMIDs are 1-9 digits."""
    return re.findall(r"\[(\d{1,9})\]", text)
