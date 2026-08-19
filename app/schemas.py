"""Service contracts — the API's public shape (drives OpenAPI docs & validation)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: int | None = Field(None, ge=1, le=20, description="Passages to ground on.")


class Source(BaseModel):
    id: str
    score: float = Field(..., description="Cross-encoder (or fused) relevance score.")
    source: str = Field(..., description="Originating document.")
    snippet: str


class AskMeta(BaseModel):
    attempts: int = Field(..., description="Retrieval passes (>1 means the re-query loop fired).")
    latency_ms: int
    model: str
    grade: str = Field(..., description="Retrieval self-grade: relevant | weak.")
    retriever: str = "hybrid(bm25+dense)+rrf+cross-encoder"


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    meta: AskMeta


class IngestRequest(BaseModel):
    documents: list["IngestDoc"]


class IngestDoc(BaseModel):
    source: str = Field(..., description="Document name / id.")
    text: str


class IngestResponse(BaseModel):
    chunks_added: int
    chunks_deduped: int
    collection_size: int


IngestRequest.model_rebuild()
