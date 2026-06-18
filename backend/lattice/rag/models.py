"""Shared RAG data models: query classes, citations, agent events, results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QueryClass(StrEnum):
    """Router output, mirroring LightRAG's dual-level retrieval insight."""

    FACTUAL = "factual"  # specific facts -> vector + card lookup (low-level)
    RELATIONAL = "relational"  # graph traversal -> neighbors, paths
    GLOBAL = "global"  # thematic synthesis -> community summaries (high-level)
    COMPARATIVE = "comparative"  # structured card diffing across papers


@dataclass
class Citation:
    """A grounded citation resolving to paper + section + evidence location."""

    marker: int  # the [n] used inline
    paper_id: str
    title: str
    section: str | None = None
    evidence_location: str | None = None
    snippet: str | None = None
    page: int | None = None  # PDF page for deep-linking, when known

    def to_json(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "evidence_location": self.evidence_location,
            "snippet": self.snippet,
            "page": self.page,
        }


@dataclass
class ChunkHit:
    chunk_id: str
    paper_id: str
    title: str
    section_title: str
    text: str
    score: float
    evidence_location: str | None = None
    page: int | None = None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class AgentEvent:
    """An SSE event streamed to the client."""

    type: str  # "status" | "tool_call" | "tool_result" | "token" | "final" | "error"
    data: dict[str, Any]


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    query_class: QueryClass = QueryClass.FACTUAL
    confidence: float = 0.0
    abstained: bool = False
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_json() for c in self.citations],
            "tool_calls": [
                {"name": t.name, "args": t.args, "error": t.error} for t in self.tool_calls
            ],
            "query_class": str(self.query_class),
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "cost_usd": round(self.cost_usd, 6),
            "tokens": {"input": self.input_tokens, "output": self.output_tokens},
        }
