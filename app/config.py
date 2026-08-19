"""Typed settings — one source of truth, env-overridable (12-factor)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # vector store
    qdrant_location: str = ":memory:"
    qdrant_collection: str = "raggraph"

    # retrieval
    top_k: int = 4
    candidate_k: int = 20
    rrf_k: int = 60
    max_retries: int = 1
    grade_min_overlap: float = 0.12  # lexical overlap floor before we re-query

    # llm
    llm_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None

    log_level: str = "INFO"


settings = Settings()
