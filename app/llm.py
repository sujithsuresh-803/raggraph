"""LLM generation adapter — Anthropic (claude-opus-5) with an offline stub fallback.

Streaming is first-class: the SSE endpoint consumes `stream()`. When no Anthropic
credentials are present we fall back to an extractive answer so the pipeline still
demonstrates end-to-end retrieval → grounded generation offline.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

from .config import settings
from .observability import LLM_TOKENS, get_logger

log = get_logger()

SYSTEM = (
    "You are a retrieval-grounded assistant. Answer ONLY from the provided context. "
    "Cite sources inline as [source]. If the context does not contain the answer, say so."
)


def _prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nGrounded answer:"


class LLM:
    def __init__(self) -> None:
        self.model = settings.llm_model
        self._client = None
        self.backend = "stub"
        if os.getenv("ANTHROPIC_API_KEY") or settings.anthropic_api_key:
            try:
                from anthropic import Anthropic  # type: ignore

                self._client = Anthropic()  # resolves key/profile automatically
                self.backend = f"anthropic:{self.model}"
            except Exception as e:  # noqa: BLE001 — optional dependency / auth
                log.info(f"llm: Anthropic unavailable, using stub ({e})")

    # ── non-streaming (used by /ask) ─────────────────────────────────
    def generate(self, context: str, question: str) -> str:
        if self._client is None:
            return self._stub(context)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(context, question)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        LLM_TOKENS.labels(backend="anthropic").inc(len(text) // 4)
        return text

    # ── streaming (used by /ask/stream, SSE) ─────────────────────────
    def stream(self, context: str, question: str) -> Iterator[str]:
        if self._client is None:
            for chunk in self._stub(context).split(" "):
                yield chunk + " "
            return
        with self._client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(context, question)}],
        ) as stream:
            for text in stream.text_stream:
                LLM_TOKENS.labels(backend="anthropic").inc(max(1, len(text) // 4))
                yield text

    # ── offline fallback ─────────────────────────────────────────────
    def _stub(self, context: str) -> str:
        first = context.strip().split("\n\n")[0][:600]
        LLM_TOKENS.labels(backend="stub").inc(len(first) // 4)
        return (
            "[offline stub — set ANTHROPIC_API_KEY for LLM-generated answers]\n"
            f"Based on the top retrieved passage:\n{first}"
        )


llm = LLM()
