"""Neo4j-backed read paths for the explorer and insight views.

In production the API and worker are separate processes, so the in-memory mirrors
in IngestionService are not shared. When a GraphReader is wired, the service's read
methods (snapshot, neighbors, claim relations, lineage) query the live graph
instead, giving cross-process correctness. Pure Cypher behind the GraphStore, so
the row->model mapping is unit-tested with a fake store and the Cypher is
exercised live in the Neo4j integration tests.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from lattice.graph.contradictions import ClaimEdge, ClaimRelation
from lattice.graph.lineage import Lineage, LineageEdge, LineageNode
from lattice.graph.models import GraphEdge, GraphNode, GraphSnapshot
from lattice.graph.store import GraphStore


class GraphReader(Protocol):
    async def snapshot(self, workspace_id: str, as_of_year: int | None = None) -> GraphSnapshot: ...
    async def neighbors(
        self, workspace_id: str, paper_id: str, min_weight: float
    ) -> list[GraphEdge]: ...
    async def claim_relations(self, workspace_id: str, relation: str | None) -> list[ClaimEdge]: ...
    async def lineage(self, workspace_id: str, method: str) -> Lineage: ...


def _components(raw: Any) -> dict[str, float]:
    if not raw:
        return {}
    data = json.loads(raw) if isinstance(raw, str) else raw
    comps = data.get("components", data) if isinstance(data, dict) else {}
    return {k: float(v) for k, v in comps.items()} if isinstance(comps, dict) else {}


class Neo4jGraphReader:
    """Reads explorer/insight views from Neo4j via a GraphStore."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    async def snapshot(self, workspace_id: str, as_of_year: int | None = None) -> GraphSnapshot:
        # Time-travel: restrict to papers published up to ``as_of_year`` (and edges
        # whose both endpoints qualify). Community/centrality stay as the precomputed
        # current values; the in-memory path recomputes them on the subgraph.
        node_rows = await self._store.execute(
            """
            MATCH (p:Paper {workspace_id: $ws})
            WHERE p.superseded_by IS NULL
              AND ($yr IS NULL OR (p.year IS NOT NULL AND p.year <= $yr))
            RETURN p.id AS id, p.title AS title, p.year AS year,
                   coalesce(p.community, 0) AS community,
                   coalesce(p.pagerank, 0.0) AS centrality,
                   coalesce(p.needs_review, false) AS needs_review
            """,
            {"ws": workspace_id, "yr": as_of_year},
        )
        edge_rows = await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws})-[r:RELATED_TO]->(b:Paper {workspace_id: $ws})
            WHERE r.invalid_at IS NULL
              AND ($yr IS NULL OR (a.year IS NOT NULL AND a.year <= $yr
                                   AND b.year IS NOT NULL AND b.year <= $yr))
            RETURN a.id AS source, b.id AS target, r.weight AS weight,
                   r.weight_components AS components
            ORDER BY r.weight DESC
            """,
            {"ws": workspace_id, "yr": as_of_year},
        )
        # Normalize centrality to [0,1] for display parity with the in-memory path.
        max_c = max((float(r["centrality"]) for r in node_rows), default=1.0) or 1.0
        nodes = [
            GraphNode(
                id=r["id"],
                title=r["title"],
                year=r["year"],
                community=int(r["community"]),
                centrality=round(float(r["centrality"]) / max_c, 4),
                needs_review=bool(r["needs_review"]),
            )
            for r in node_rows
        ]
        edges = [
            GraphEdge(
                source=r["source"],
                target=r["target"],
                weight=round(float(r["weight"]), 4),
                components=_components(r["components"]),
            )
            for r in edge_rows
        ]
        return GraphSnapshot(nodes=nodes, edges=edges)

    async def neighbors(
        self, workspace_id: str, paper_id: str, min_weight: float = 0.0
    ) -> list[GraphEdge]:
        rows = await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws, id: $pid})-[r:RELATED_TO]-(b:Paper)
            WHERE r.invalid_at IS NULL AND r.weight >= $minw
            RETURN $pid AS source, b.id AS target, r.weight AS weight,
                   r.weight_components AS components
            ORDER BY r.weight DESC
            """,
            {"ws": workspace_id, "pid": paper_id, "minw": min_weight},
        )
        return [
            GraphEdge(
                r["source"], r["target"], round(float(r["weight"]), 4), _components(r["components"])
            )
            for r in rows
        ]

    async def claim_relations(
        self, workspace_id: str, relation: str | None = None
    ) -> list[ClaimEdge]:
        rel_filter = ""
        if relation:
            rel_filter = (
                f":{relation}" if relation in ("SUPPORTS", "CONTRADICTS", "EXTENDS") else ""
            )
        rows = await self._store.execute(
            f"""
            MATCH (a:Claim {{workspace_id: $ws}})-[r{rel_filter}]->(b:Claim)
            WHERE type(r) IN ['SUPPORTS','CONTRADICTS','EXTENDS']
            MATCH (pa:Paper)-[:MAKES_CLAIM]->(a)
            MATCH (pb:Paper)-[:MAKES_CLAIM]->(b)
            RETURN a.id AS sid, b.id AS tid, pa.id AS sp, pb.id AS tp,
                   type(r) AS relation, coalesce(r.confidence, 0.0) AS confidence,
                   a.text AS stext, b.text AS ttext,
                   a.evidence_location AS sev, b.evidence_location AS tev
            """,
            {"ws": workspace_id},
        )
        return [
            ClaimEdge(
                source_id=r["sid"],
                target_id=r["tid"],
                source_paper=r["sp"],
                target_paper=r["tp"],
                relation=ClaimRelation(r["relation"]),
                confidence=float(r["confidence"]),
                source_text=r["stext"] or "",
                target_text=r["ttext"] or "",
                source_evidence=r["sev"] or "",
                target_evidence=r["tev"] or "",
            )
            for r in rows
        ]

    async def lineage(self, workspace_id: str, method: str) -> Lineage:
        node_rows = await self._store.execute(
            """
            MATCH (p:Paper {workspace_id: $ws})-[:USES_METHOD]->(m:Method)
            WHERE toLower(m.name) = toLower($method) AND p.superseded_by IS NULL
            RETURN p.id AS id, p.title AS title, p.year AS year
            """,
            {"ws": workspace_id, "method": method},
        )
        members = {r["id"] for r in node_rows}
        nodes = [
            LineageNode(paper_id=r["id"], title=r["title"], year=r["year"], methods={method})
            for r in node_rows
        ]
        nodes.sort(key=lambda n: (n.year is None, n.year or 0, n.title))
        cite_rows = await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws})-[:CITES]->(b:Paper {workspace_id: $ws})
            RETURN a.id AS citing, b.id AS cited
            """,
            {"ws": workspace_id},
        )
        edges: list[LineageEdge] = []
        seen: set[tuple[str, str]] = set()
        by_id = {n.paper_id: n for n in nodes}
        for r in cite_rows:
            citing, cited = r["citing"], r["cited"]
            if citing not in members or cited not in members:
                continue
            a, b = by_id[cited], by_id[citing]
            older, newer = (
                (a.paper_id, b.paper_id)
                if (a.year or 0) <= (b.year or 0)
                else (b.paper_id, a.paper_id)
            )
            if (older, newer) in seen:
                continue
            seen.add((older, newer))
            edges.append(LineageEdge(source=older, target=newer, kind="cites"))
        return Lineage(method=method, nodes=nodes, edges=edges)
