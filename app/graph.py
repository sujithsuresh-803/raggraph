"""LangGraph orchestration.

    START → retrieve → grade ─┬─(relevant)→ generate → END
                              └─(weak, retries left)→ rewrite → retrieve …

The conditional edge after `grade` is the agentic bit: when retrieval looks weak
the graph rewrites the query and tries again, up to MAX_RETRIES.
"""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import settings
from .llm import llm
from .observability import get_logger
from .retrieval import store, tokens

log = get_logger()

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "which", "how", "why", "who", "when", "does", "do", "did",
    "can", "with", "about", "me", "my", "i", "you", "it", "this", "that",
}


class RAGState(TypedDict, total=False):
    question: str
    query: str            # possibly-rewritten retrieval query
    passages: list[dict]
    context: str
    answer: str
    grade: str            # "relevant" | "weak"
    attempts: int


def _format(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)


def _overlap(question: str, passages: list[dict]) -> float:
    q = set(tokens(question)) - _STOP
    if not q or not passages:
        return 0.0
    joined = set(tokens(" ".join(p["text"] for p in passages)))
    return len(q & joined) / len(q)


# ── nodes ─────────────────────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    top_k = state.get("top_k", settings.top_k)
    passages = store.retrieve(state.get("query") or state["question"], top_k)
    return {"passages": passages, "attempts": state.get("attempts", 0) + 1}


def grade_node(state: RAGState) -> RAGState:
    grade = "relevant" if _overlap(state["question"], state["passages"]) >= settings.grade_min_overlap else "weak"
    return {"grade": grade}


def rewrite_node(state: RAGState) -> RAGState:
    """Offline query rewrite: drop stopwords, keep salient keywords."""
    kws = [t for t in tokens(state["question"]) if t not in _STOP]
    return {"query": " ".join(kws) or state["question"]}


def generate_node(state: RAGState) -> RAGState:
    context = _format(state["passages"])
    return {"context": context, "answer": llm.generate(context, state["question"])}


def route_after_grade(state: RAGState) -> str:
    if state["grade"] == "relevant" or state["attempts"] > settings.max_retries:
        return "generate"
    return "rewrite"


def _build() -> "CompiledStateGraph":  # noqa: F821
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


GRAPH = _build()


def run(question: str, top_k: int | None = None) -> RAGState:
    """Full graph invocation — used by the JSON /ask endpoint."""
    init: RAGState = {"question": question, "attempts": 0}
    if top_k:
        init["top_k"] = top_k
    return GRAPH.invoke(init)


def build_context(question: str, top_k: int | None = None) -> RAGState:
    """Run retrieve → grade → (rewrite → retrieve) loop only, for streamed generation."""
    state: RAGState = {"question": question, "attempts": 0}
    if top_k:
        state["top_k"] = top_k
    while True:
        state.update(retrieve_node(state))
        state.update(grade_node(state))
        if route_after_grade(state) == "generate":
            break
        state.update(rewrite_node(state))
    state["context"] = _format(state["passages"])
    return state
