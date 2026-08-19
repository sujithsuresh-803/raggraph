"""Hybrid retrieval over a vector store.

Pipeline:  dense (Qdrant) + sparse (BM25)  →  Reciprocal Rank Fusion  →  cross-encoder re-rank.
This is the 'vector store / hybrid / re-ranking' surface the JD asks for.
"""
from __future__ import annotations

import re
import threading

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import Embedder, Reranker
from .observability import get_logger

log = get_logger()
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Store:
    """Vector store + keyword index + fusion + re-ranking, behind one interface."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.collection = settings.qdrant_collection
        self.client = self._connect()
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        self._next_id = 0
        self.ids: list[int] = []
        self.corpus_tokens: list[list[str]] = []
        self.registry: dict[int, dict] = {}   # id -> {text, source}
        self._bm25: BM25Okapi | None = None

    def _connect(self) -> QdrantClient:
        loc = settings.qdrant_location
        if loc.startswith("http"):
            log.info(f"qdrant: connecting to server {loc}")
            return QdrantClient(url=loc)
        if loc == ":memory:":
            return QdrantClient(location=":memory:")
        return QdrantClient(path=loc)

    @property
    def size(self) -> int:
        return len(self.ids)

    # ── writes ────────────────────────────────────────────────────────
    def add(self, chunks: list[dict]) -> None:
        """chunks: [{text, source, vector}] — vectors pre-computed by the ingester."""
        with self._lock:
            points = []
            for c in chunks:
                cid = self._next_id
                self._next_id += 1
                self.ids.append(cid)
                self.corpus_tokens.append(tokens(c["text"]))
                self.registry[cid] = {"text": c["text"], "source": c["source"]}
                points.append(
                    PointStruct(id=cid, vector=c["vector"], payload={"source": c["source"]})
                )
            if points:
                self.client.upsert(collection_name=self.collection, points=points)
                self._bm25 = BM25Okapi(self.corpus_tokens)  # rebuild keyword index

    # ── reads ─────────────────────────────────────────────────────────
    def _dense(self, query: str, k: int) -> list[int]:
        vec = self.embedder.encode([query])[0].tolist()
        res = self.client.query_points(collection_name=self.collection, query=vec, limit=k)
        return [int(p.id) for p in res.points]

    def max_similarity(self, vector: list[float]) -> float:
        """Top-1 cosine of a vector against the index — used for near-dup detection."""
        if self.size == 0:
            return 0.0
        res = self.client.query_points(collection_name=self.collection, query=vector, limit=1)
        return float(res.points[0].score) if res.points else 0.0

    def _sparse(self, query: str, k: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:k]]

    def _rrf(self, *rankings: list[int]) -> list[int]:
        """Reciprocal Rank Fusion — order-based, scale-free merge of retrievers."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank)
        return sorted(fused, key=lambda c: fused[c], reverse=True)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Full hybrid retrieve → fuse → cross-encoder re-rank; returns top_k passages."""
        if self.size == 0:
            return []
        ck = settings.candidate_k
        fused = self._rrf(self._dense(query, ck), self._sparse(query, ck))[:ck]
        passages = [self.registry[c]["text"] for c in fused]
        scores = self.reranker.score(query, passages)
        order = sorted(range(len(fused)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": str(fused[i]),
                "source": self.registry[fused[i]]["source"],
                "text": self.registry[fused[i]]["text"],
                "score": round(float(scores[i]), 4),
            }
            for i in order
        ]


store = Store()
