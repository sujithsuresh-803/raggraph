"""Ingestion: chunk → embed → dedup → index.

Chunking is sentence-aware with overlap; dedup catches both exact repeats
(content hash) and near-duplicates (cosine similarity of embeddings).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .observability import get_logger
from .retrieval import store

log = get_logger()

_SENT = re.compile(r"(?<=[.!?])\s+")
NEAR_DUP_COSINE = 0.97


def chunk(text: str, target_chars: int = 700, overlap_chars: int = 120) -> list[str]:
    """Greedy sentence packing up to ~target size, with a small trailing overlap."""
    sentences = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > target_chars:
            chunks.append(buf.strip())
            buf = (buf[-overlap_chars:] + " " + s) if overlap_chars else s
        else:
            buf = f"{buf} {s}".strip()
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _norm_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def ingest(documents: list[dict]) -> dict:
    """documents: [{source, text}] → returns counts. Dedups within-batch + against store."""
    seen_hashes = {_norm_hash(v["text"]) for v in store.registry.values()}
    kept_vectors: list[np.ndarray] = []

    pending: list[tuple[str, str]] = []  # (text, source)
    for doc in documents:
        for ch in chunk(doc["text"]):
            pending.append((ch, doc["source"]))

    if not pending:
        return {"chunks_added": 0, "chunks_deduped": 0, "collection_size": store.size}

    vectors = store.embedder.encode([t for t, _ in pending])
    added, deduped = [], 0
    for (text, source), vec in zip(pending, vectors):
        h = _norm_hash(text)
        if h in seen_hashes:                                    # exact repeat
            deduped += 1
            continue
        within = float(np.max(np.stack(kept_vectors) @ vec)) if kept_vectors else 0.0
        if within >= NEAR_DUP_COSINE or store.max_similarity(vec.tolist()) >= NEAR_DUP_COSINE:
            deduped += 1                                         # near-dup (this batch or index)
            continue
        seen_hashes.add(h)
        kept_vectors.append(vec)
        added.append({"text": text, "source": source, "vector": vec.tolist()})

    store.add(added)
    log.info(f"ingest: +{len(added)} chunks, {deduped} deduped, size={store.size}")
    return {"chunks_added": len(added), "chunks_deduped": deduped, "collection_size": store.size}
