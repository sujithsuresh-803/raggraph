"""Smoke test — runs fully offline (hashing embeddings + stub LLM, no downloads/keys)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # context manager fires the lifespan (seeds the index)
        yield c


def test_health_and_root(client):
    assert client.get("/healthz").json()["status"] == "ok"
    body = client.get("/").json()
    assert body["service"] == "RAGgraph"
    assert body["indexed_chunks"] > 0          # sample docs seeded on startup


def test_ingest_and_dedup(client):
    doc = {"documents": [{"source": "note.md", "text": "Alpha beta gamma. " * 40}]}
    first = client.post("/ingest", json=doc).json()
    assert first["chunks_added"] >= 1
    second = client.post("/ingest", json=doc).json()   # identical → all deduped
    assert second["chunks_added"] == 0
    assert second["chunks_deduped"] >= 1


def test_ask_returns_grounded_sources(client):
    r = client.post("/ask", json={"question": "What is reciprocal rank fusion?"})
    assert r.status_code == 200
    data = r.json()
    assert data["sources"], "expected retrieved sources"
    assert data["meta"]["retriever"].startswith("hybrid")
    assert "rank" in " ".join(s["snippet"].lower() for s in data["sources"])


def test_stream_emits_sse_events(client):
    r = client.post("/ask/stream", json={"question": "How does LangGraph branch?"})
    assert r.status_code == 200
    assert "event: meta" in r.text
    assert "event: token" in r.text
    assert "event: done" in r.text


def test_metrics_exposed(client):
    body = client.get("/metrics").text
    assert "raggraph_requests_total" in body
