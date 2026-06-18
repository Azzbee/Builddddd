"""Typed tools exposed to the RAG agent.

Each tool reads from an injected store and returns JSON-serializable results with
enough provenance (paper id, section, evidence location) for the synthesizer to
build grounded citations. Stores are protocols so the toolbox is testable with
fakes and so the same tools back the MCP server (extra #8).
"""

from __future__ import annotations

from typing import Any, Protocol

from lattice.core.errors import CypherValidationError
from lattice.embeddings.base import TextEmbedder
from lattice.graph.store import GraphStore, validate_read_only
from lattice.rag.models import ChunkHit


class VectorStore(Protocol):
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkHit]: ...


class CardStore(Protocol):
    async def get_card(self, paper_id: str) -> dict[str, Any] | None: ...


# Tool JSON schemas advertised to the LLM (provider-agnostic descriptions).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_chunks",
        "description": "Hybrid vector + keyword search over paper text. Use for specific facts.",
        "parameters": {"query": "string", "k": "int (optional)"},
    },
    {
        "name": "get_paper_card",
        "description": "Fetch the full structured card for a paper by id.",
        "parameters": {"paper_id": "string"},
    },
    {
        "name": "graph_neighbors",
        "description": "Typed graph traversal from a paper (RELATED_TO/CITES/etc).",
        "parameters": {
            "paper_id": "string",
            "edge_types": "list[string] (optional)",
            "min_weight": "float (optional)",
            "hops": "int <=2 (optional)",
        },
    },
    {
        "name": "find_papers",
        "description": "Structured filter for papers by method/dataset/concept/year range.",
        "parameters": {
            "method": "string?",
            "dataset": "string?",
            "concept": "string?",
            "year_from": "int?",
            "year_to": "int?",
        },
    },
    {
        "name": "compare_papers",
        "description": "Aligned comparison of several papers across card dimensions.",
        "parameters": {"paper_ids": "list[string]", "dimensions": "list[string] (optional)"},
    },
    {
        "name": "run_cypher",
        "description": "Run a read-only Cypher query (validated, LIMIT enforced).",
        "parameters": {"query": "string"},
    },
    {
        "name": "find_contradictions",
        "description": "Find Claim-level CONTRADICTS edges, optionally for a concept.",
        "parameters": {"concept": "string (optional)"},
    },
]


class Toolbox:
    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorStore,
        cards: CardStore,
        embedder: TextEmbedder,
        *,
        workspace_id: str = "default",
        top_k: int = 12,
        cypher_limit: int = 200,
    ) -> None:
        self._graph = graph
        self._vectors = vectors
        self._cards = cards
        self._embedder = embedder
        self._ws = workspace_id
        self._top_k = top_k
        self._cypher_limit = cypher_limit

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        return await handler(args)

    # ------------------------------------------------------------------ tools
    async def _tool_search_chunks(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        k = int(args.get("k", self._top_k))
        emb = self._embedder.embed([query])[0]
        hits = await self._vectors.hybrid_search(query, emb, k, {"workspace_id": self._ws})
        return {
            "chunks": [
                {
                    "paper_id": h.paper_id,
                    "title": h.title,
                    "section": h.section_title,
                    "text": h.text,
                    "score": round(h.score, 4),
                    "evidence_location": h.evidence_location or h.section_title,
                    "page": h.page,
                }
                for h in hits
            ]
        }

    async def _tool_get_paper_card(self, args: dict[str, Any]) -> dict[str, Any]:
        card = await self._cards.get_card(str(args["paper_id"]))
        return {"card": card} if card else {"error": "not_found"}

    async def _tool_graph_neighbors(self, args: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(args["paper_id"])
        edge_types = args.get("edge_types") or ["RELATED_TO"]
        min_weight = float(args.get("min_weight", 0.0))
        hops = min(int(args.get("hops", 1)), 2)
        rel = "|".join(re_safe(t) for t in edge_types)
        rows = await self._graph.execute(
            f"""
            MATCH (p:Paper {{workspace_id: $ws, id: $pid}})-[r:{rel}*1..{hops}]->(n:Paper)
            WITH n, r, last(r) AS edge
            WHERE coalesce(edge.weight, 1.0) >= $minw
              AND (edge.invalid_at IS NULL)
            RETURN DISTINCT n.id AS id, n.title AS title,
                   coalesce(edge.weight, 1.0) AS weight
            ORDER BY weight DESC
            LIMIT $limit
            """,
            {"ws": self._ws, "pid": paper_id, "minw": min_weight, "limit": self._cypher_limit},
        )
        return {"neighbors": rows}

    async def _tool_find_papers(self, args: dict[str, Any]) -> dict[str, Any]:
        clauses = ["p.workspace_id = $ws"]
        params: dict[str, Any] = {"ws": self._ws, "limit": self._cypher_limit}
        match = "MATCH (p:Paper)"
        if args.get("method"):
            match += "-[:USES_METHOD]->(m:Method)"
            clauses.append("toLower(m.name) CONTAINS toLower($method)")
            params["method"] = args["method"]
        if args.get("dataset"):
            match += "\nMATCH (p)-[:USES_DATASET]->(d:Dataset)"
            clauses.append("toLower(d.name) CONTAINS toLower($dataset)")
            params["dataset"] = args["dataset"]
        if args.get("concept"):
            match += "\nMATCH (p)-[:ADDRESSES]->(c:Concept)"
            clauses.append("toLower(c.name) CONTAINS toLower($concept)")
            params["concept"] = args["concept"]
        if args.get("year_from") is not None:
            clauses.append("p.year >= $year_from")
            params["year_from"] = int(args["year_from"])
        if args.get("year_to") is not None:
            clauses.append("p.year <= $year_to")
            params["year_to"] = int(args["year_to"])
        where = " AND ".join(clauses)
        rows = await self._graph.execute(
            f"{match}\nWHERE {where}\n"
            "RETURN DISTINCT p.id AS id, p.title AS title, p.year AS year\n"
            "ORDER BY p.year DESC\nLIMIT $limit",
            params,
        )
        return {"papers": rows}

    async def _tool_compare_papers(self, args: dict[str, Any]) -> dict[str, Any]:
        ids = list(args.get("paper_ids") or [])
        dims = args.get("dimensions") or [
            "problem_statement",
            "methodology",
            "datasets",
            "key_results",
            "limitations",
        ]
        cards = []
        for pid in ids:
            card = await self._cards.get_card(str(pid))
            if card:
                cards.append({"paper_id": pid, **{d: card.get(d) for d in dims}})
        return {"dimensions": dims, "papers": cards}

    async def _tool_run_cypher(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            safe = validate_read_only(str(args["query"]), self._cypher_limit)
        except CypherValidationError as exc:
            return {"error": str(exc)}
        rows = await self._graph.execute(safe, {"ws": self._ws})
        return {"rows": rows}

    async def _tool_find_contradictions(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"ws": self._ws, "limit": self._cypher_limit}
        concept_match = ""
        if args.get("concept"):
            concept_match = (
                "MATCH (p1)-[:ADDRESSES]->(c:Concept) "
                "WHERE toLower(c.name) CONTAINS toLower($concept) "
            )
            params["concept"] = args["concept"]
        rows = await self._graph.execute(
            f"""
            MATCH (a:Claim {{workspace_id: $ws}})-[:CONTRADICTS]->(b:Claim)
            MATCH (p1:Paper)-[:MAKES_CLAIM]->(a)
            MATCH (p2:Paper)-[:MAKES_CLAIM]->(b)
            {concept_match}
            RETURN a.text AS claim_a, a.evidence_location AS ev_a, p1.id AS paper_a,
                   b.text AS claim_b, b.evidence_location AS ev_b, p2.id AS paper_b
            LIMIT $limit
            """,
            params,
        )
        return {"contradictions": rows}


def re_safe(token: str) -> str:
    """Whitelist edge-type identifiers to prevent Cypher injection in labels."""
    cleaned = "".join(ch for ch in token if ch.isalnum() or ch == "_")
    if not cleaned:
        raise ValueError("invalid edge type")
    return cleaned
