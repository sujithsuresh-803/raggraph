"""Embeddings + cross-encoder re-ranker, both with graceful offline fallbacks.

Preferred: sentence-transformers (a real bi-encoder + cross-encoder).
Fallback:  a dependency-free hashing embedding and a lexical re-ranker, so the
service still runs (and tests pass) on a minimal install with no model download.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from .observability import get_logger

log = get_logger()
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ── dense embedder ───────────────────────────────────────────────────
class Embedder:
    """Bi-encoder. Uses sentence-transformers if available, else hashing."""

    def __init__(self) -> None:
        self.backend = "hash"
        self.dim = 384
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dim = self._model.get_sentence_embedding_dimension()
            self.backend = "sentence-transformers/all-MiniLM-L6-v2"
        except Exception as e:  # noqa: BLE001 — optional dependency
            log.info(f"embedder: falling back to hashing embeddings ({e})")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(
                self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
            )
        return np.vstack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-hashed-tokens vector, L2-normalized."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        n = math.sqrt(float(vec @ vec)) or 1.0
        return vec / n


# ── cross-encoder re-ranker ──────────────────────────────────────────
class Reranker:
    """Re-scores (query, passage) pairs jointly — the accuracy step after fusion."""

    def __init__(self) -> None:
        self.backend = "lexical-overlap"
        self._model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.backend = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        except Exception as e:  # noqa: BLE001 — optional dependency
            log.info(f"reranker: falling back to lexical overlap ({e})")

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is not None:
            return [float(s) for s in self._model.predict([(query, p) for p in passages])]
        q = set(_tokens(query))
        out = []
        for p in passages:
            toks = _tokens(p)
            overlap = sum(1 for t in toks if t in q)
            out.append(overlap / (len(toks) ** 0.5 + 1e-9))
        return out
