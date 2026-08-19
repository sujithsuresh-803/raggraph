"""FastAPI service — typed contracts, SSE streaming, Prometheus metrics, probes."""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .graph import build_context, run
from .ingest import ingest
from .llm import llm
from .observability import (
    LATENCY,
    REQUESTS,
    RETRIEVAL_ATTEMPTS,
    get_logger,
    log,
    new_trace_id,
)
from .retrieval import store
from .schemas import AskMeta, AskRequest, AskResponse, IngestRequest, IngestResponse, Source

logger = get_logger()
_ROOT = Path(__file__).resolve().parent.parent          # repo root (holds sample_docs/, scripts/)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if store.size == 0:                                  # warm the index on boot
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from scripts.seed import seed_from_dir

        seed_from_dir(str(_ROOT / "sample_docs"))
    yield


app = FastAPI(
    title="RAGgraph",
    version=__version__,
    description="Agentic RAG microservice — hybrid retrieval + re-ranking, "
    "LangGraph orchestration, streaming, and Prometheus observability.",
    lifespan=lifespan,
)


@app.middleware("http")
async def trace_and_meter(request: Request, call_next):
    tid = new_trace_id()
    route = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        status_label = locals().get("status", 500)
        LATENCY.labels(route=route).observe(elapsed)
        REQUESTS.labels(route=route, status=str(status_label)).inc()
        log(logger, 20, "request", route=route, status=status_label,
            latency_ms=round(elapsed * 1000), trace_id=tid)


@app.get("/")
def root() -> dict:
    return {
        "service": "RAGgraph",
        "version": __version__,
        "vector_store": store.client.__class__.__name__,
        "embedder": store.embedder.backend,
        "reranker": store.reranker.backend,
        "llm": llm.backend,
        "indexed_chunks": store.size,
        "endpoints": ["/ask", "/ask/stream", "/ingest", "/healthz", "/readyz", "/metrics", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if store.size == 0:
        return JSONResponse({"status": "not-ready", "reason": "index empty"}, status_code=503)
    return {"status": "ready", "indexed_chunks": store.size}


@app.get("/metrics")
def metrics() -> StreamingResponse:
    return StreamingResponse(iter([generate_latest()]), media_type=CONTENT_TYPE_LATEST)


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    result = ingest([d.model_dump() for d in req.documents])
    return IngestResponse(**result)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    state = run(req.question, req.top_k)
    RETRIEVAL_ATTEMPTS.observe(state.get("attempts", 1))
    sources = [
        Source(id=p["id"], score=p["score"], source=p["source"], snippet=p["text"][:240])
        for p in state.get("passages", [])
    ]
    return AskResponse(
        answer=state.get("answer", ""),
        sources=sources,
        meta=AskMeta(
            attempts=state.get("attempts", 1),
            latency_ms=round((time.perf_counter() - t0) * 1000),
            model=llm.backend,
            grade=state.get("grade", "relevant"),
        ),
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events: retrieval meta first, then streamed answer tokens, then done."""
    def events() -> Iterator[str]:
        t0 = time.perf_counter()
        state = build_context(req.question, req.top_k)
        RETRIEVAL_ATTEMPTS.observe(state.get("attempts", 1))
        sources = [
            {"id": p["id"], "score": p["score"], "source": p["source"], "snippet": p["text"][:240]}
            for p in state["passages"]
        ]
        yield _sse("meta", {"attempts": state["attempts"], "grade": state["grade"],
                            "model": llm.backend, "sources": sources})
        for token in llm.stream(state["context"], req.question):
            yield _sse("token", {"text": token})
        yield _sse("done", {"latency_ms": round((time.perf_counter() - t0) * 1000)})

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
