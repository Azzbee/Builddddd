"""Query router: classify a question to steer retrieval (LightRAG dual-level).

A fast, deterministic heuristic handles the common cases; an optional LLM
classifier can be plugged in for ambiguous ones. The heuristic alone is good
enough to route the bulk of queries and keeps latency and cost down.
"""

from __future__ import annotations

import re

from lattice.rag.models import QueryClass

_COMPARATIVE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|differ|contrast)\b", re.I
)
_RELATIONAL = re.compile(
    r"\b(build on|builds on|extends?|related to|connect|cite[sd]?|citing|"
    r"influenced|lineage|neighbou?rs?|downstream|upstream|who (?:cites|uses))\b",
    re.I,
)
_GLOBAL = re.compile(
    r"\b(overall|landscape|themes?|schools? of thought|main (?:ideas|approaches|trends)|"
    r"summar(?:y|ize|ise)|state of the (?:art|field)|big picture|across the (?:corpus|field)|"
    r"what are the (?:main|major)|clusters?)\b",
    re.I,
)


def classify_heuristic(query: str) -> tuple[QueryClass, float]:
    """Classify a query with regex heuristics. Returns (class, confidence)."""
    q = query.strip()
    if _COMPARATIVE.search(q):
        return QueryClass.COMPARATIVE, 0.8
    if _GLOBAL.search(q):
        return QueryClass.GLOBAL, 0.75
    if _RELATIONAL.search(q):
        return QueryClass.RELATIONAL, 0.75
    return QueryClass.FACTUAL, 0.6


def classify(query: str) -> QueryClass:
    return classify_heuristic(query)[0]
