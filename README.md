# 🕸️ RAGgraph — an agentic RAG microservice

A production-shaped Retrieval-Augmented Generation service: **hybrid retrieval +
cross-encoder re-ranking**, **LangGraph** orchestration with a self-correcting
re-query loop, **streaming** answers, first-class **observability**, and
**Docker + Kubernetes** deployment.

It runs **fully offline** out of the box (local hashing embeddings, a lexical
re-ranker, and an extractive stub LLM) — add `sentence-transformers` and an
`ANTHROPIC_API_KEY` to upgrade quality with zero code changes.

---

## Why this exists — skills it demonstrates

| Capability | Where it lives |
|---|---|
| **Docker / Kubernetes** | [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml) (app + Qdrant), [`k8s/`](k8s/) (Deployment with liveness/readiness probes, Service, resource limits, Secret-mounted key) |
| **LangGraph orchestration** | [`app/graph.py`](app/graph.py) — a `StateGraph` with a **conditional edge** that re-queries when retrieval grades "weak" |
| **Vector stores / hybrid / re-ranking** | [`app/retrieval.py`](app/retrieval.py) — Qdrant vector search **+** BM25, fused with **Reciprocal Rank Fusion**, then a **cross-encoder** re-ranker; [`app/ingest.py`](app/ingest.py) — sentence-aware **chunking** + exact & near-duplicate **dedup** |
| **Platform / infra at scale** | [`app/main.py`](app/main.py) — typed **service contracts** + OpenAPI, **SSE token streaming**; [`app/observability.py`](app/observability.py) — structured JSON logs w/ trace ids, **Prometheus `/metrics`**, `/healthz` + `/readyz` probes |

---

## Architecture

```
                 ┌──────────── LangGraph ────────────┐
  question ──▶ retrieve ──▶ grade ──(relevant)──▶ generate ──▶ answer + sources
                 ▲            │
                 │         (weak,                 hybrid retrieval:
              rewrite ◀── retries left)             dense (Qdrant) + BM25
                                                    → RRF fusion
                                                    → cross-encoder re-rank
```

- **Ingest**: chunk → embed → dedup (content hash + embedding cosine) → upsert to Qdrant + BM25 index.
- **Retrieve**: dense + sparse candidates → RRF → cross-encoder re-rank → top-k.
- **Orchestrate**: LangGraph grades the context and loops back through a query rewrite if it's weak.
- **Generate**: Claude (`claude-opus-5`) streams a grounded, cited answer over SSE.

## Quickstart (local, no keys, no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload      # seeds sample_docs/ on startup
```

```bash
# ask (JSON)
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is reciprocal rank fusion?"}' | python -m json.tool

# ask (streamed tokens over SSE)
curl -N localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question":"How does LangGraph branch on state?"}'

# add your own docs
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"source":"my.md","text":"..."}]}'
```

Interactive docs at `http://localhost:8000/docs`, metrics at `/metrics`.

### Upgrade to real models
```bash
pip install sentence-transformers anthropic
export ANTHROPIC_API_KEY=sk-ant-...        # LLM-generated answers instead of the stub
```

### Run with Docker (app + Qdrant server)
```bash
docker compose up --build
```

### Deploy to Kubernetes
```bash
docker build -t <registry>/raggraph:latest . && docker push <registry>/raggraph:latest
kubectl apply -f k8s/
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | Grounded answer + ranked sources + meta (attempts, grade, latency) |
| `POST` | `/ask/stream` | Same, streamed as SSE (`meta` → `token`… → `done`) |
| `POST` | `/ingest` | Chunk + dedup + index documents |
| `GET` | `/healthz` · `/readyz` | Liveness · readiness (503 until the index is warm) |
| `GET` | `/metrics` | Prometheus exposition |

## Design notes

- **Graceful degradation is deliberate** — every heavy dependency (embeddings, re-ranker,
  LLM) has an offline fallback, so the service and its tests run on a minimal install.
  Swapping in the real model is an env/dependency change, not a code change.
- **Qdrant runs in-process (`:memory:`) or against a server** via one env var — the same
  code path demonstrates the vector-store API locally and scales out in compose/k8s.
- **RRF over score-normalization** — fusing by rank avoids having to calibrate BM25 and
  cosine onto a shared scale.

## Tests
```bash
pip install pytest && pytest -q      # runs offline; no keys or model downloads
```

---
*Built by Sujith Suresh.*
