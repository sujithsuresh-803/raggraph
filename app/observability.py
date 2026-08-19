"""Observability: structured JSON logs, request tracing, Prometheus metrics.

This is the 'platform at scale' surface — every request carries a trace id,
emits a structured log line, and updates latency/counter metrics scraped at /metrics.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from prometheus_client import Counter, Histogram

from .config import settings

# ── trace id propagated through the request lifecycle ────────────────
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:12]
    trace_id_var.set(tid)
    return tid


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "raggraph") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonLogFormatter())
        logger.addHandler(h)
        logger.setLevel(settings.log_level.upper())
        logger.propagate = False
    return logger


def log(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


# ── Prometheus metrics ───────────────────────────────────────────────
REQUESTS = Counter("raggraph_requests_total", "Requests by route & status", ["route", "status"])
LATENCY = Histogram("raggraph_request_latency_seconds", "Request latency", ["route"])
RETRIEVAL_ATTEMPTS = Histogram(
    "raggraph_retrieval_attempts", "Retrieval passes per query", buckets=[1, 2, 3, 4]
)
LLM_TOKENS = Counter("raggraph_llm_output_tokens_total", "Approx LLM output tokens", ["backend"])
