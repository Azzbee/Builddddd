"""Lineage view: the temporal DAG of a method family.

Given papers that use a method and the directed citation edges among them, build a
time-ordered DAG ("how transformer-based forecasting evolved"). Citation edges are
oriented to point from older to newer work where years are known, so the result
reads as a lineage rather than a tangle. Pure and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice.core.hashing import normalize_text


@dataclass
class LineageNode:
    paper_id: str
    title: str
    year: int | None
    methods: set[str]


@dataclass
class LineageEdge:
    source: str  # older
    target: str  # newer
    kind: str  # "cites" | "extends"

    def to_json(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "kind": self.kind}


@dataclass
class Lineage:
    method: str
    nodes: list[LineageNode] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        timeline: dict[int, list[str]] = {}
        for n in self.nodes:
            timeline.setdefault(n.year or 0, []).append(n.paper_id)
        return {
            "method": self.method,
            "nodes": [
                {"id": n.paper_id, "title": n.title, "year": n.year} for n in self.nodes
            ],
            "edges": [e.to_json() for e in self.edges],
            "timeline": {str(y): ids for y, ids in sorted(timeline.items())},
        }


def build_lineage(
    nodes: list[LineageNode],
    cite_edges: list[tuple[str, str]],
    method: str,
    *,
    extends_edges: list[tuple[str, str]] | None = None,
) -> Lineage:
    """Build the lineage DAG for a method family.

    ``cite_edges`` are (citing, cited) pairs. Edges are reoriented older->newer by
    year so the DAG is acyclic and reads chronologically; ties keep the original
    citation direction (citing is usually the newer work, so we flip to cited->citing).
    """
    target = normalize_text(method)
    members = {n.paper_id: n for n in nodes if target in {normalize_text(m) for m in n.methods}}
    ordered = sorted(members.values(), key=lambda n: (n.year is None, n.year or 0, n.title))

    edges: list[LineageEdge] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, kind: str) -> None:
        if a not in members or b not in members or a == b:
            return
        na, nb = members[a], members[b]
        # Orient older -> newer.
        if (na.year or 0) <= (nb.year or 0):
            older, newer = a, b
        else:
            older, newer = b, a
        key = (older, newer, kind)
        if key[:2] in seen:
            return
        seen.add(key[:2])
        edges.append(LineageEdge(source=older, target=newer, kind=kind))

    for citing, cited in cite_edges:
        add(cited, citing, "cites")  # cited is typically older -> newer citing
    for a, b in extends_edges or []:
        add(a, b, "extends")

    return Lineage(method=method, nodes=ordered, edges=edges)
