"""Shared graph view-models for the explorer (used by the service and reader)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    components: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "components": self.components,
        }


@dataclass
class GraphNode:
    id: str
    title: str
    year: int | None
    community: int
    centrality: float
    needs_review: bool

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "community": self.community,
            "centrality": self.centrality,
            "needs_review": self.needs_review,
        }


@dataclass
class GraphSnapshot:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
