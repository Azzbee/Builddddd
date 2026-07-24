"""Idempotent graph writer. Every write is a MERGE so re-ingestion is a no-op.

All nodes and edges carry ``workspace_id`` (multi-tenant ready). RELATED_TO edges
are bi-temporal: they carry ``valid_from`` / ``invalid_at`` and a JSON
``weight_components`` blob, and are superseded (invalidated) rather than deleted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from lattice.core.hashing import normalize_text, stable_id
from lattice.extraction.schemas import PaperCard
from lattice.graph.evolution import EdgeUpdate
from lattice.graph.store import GraphStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


class GraphWriter:
    def __init__(self, store: GraphStore, workspace_id: str = "default") -> None:
        self._store = store
        self._ws = workspace_id

    # ------------------------------------------------------------------ nodes
    async def upsert_paper(self, card: PaperCard) -> None:
        await self._store.execute(
            """
            MERGE (p:Paper {workspace_id: $ws, id: $id})
            SET p.title = $title, p.year = $year, p.doi = $doi, p.arxiv_id = $arxiv_id,
                p.s2_paper_id = $s2, p.venue = $venue, p.paper_type = $ptype,
                p.confidence = $conf, p.needs_review = $needs_review,
                p.problem_statement = $problem, p.updated_at = $now,
                p.created_at = coalesce(p.created_at, $now),
                p.domains = $domains
            """,
            {
                "ws": self._ws,
                "id": card.paper_id,
                "title": card.title,
                "year": card.year,
                "doi": card.doi,
                "arxiv_id": card.arxiv_id,
                "s2": card.s2_paper_id,
                "venue": card.venue,
                "ptype": str(card.paper_type),
                "conf": card.confidence,
                "needs_review": card.needs_review,
                "problem": card.problem_statement,
                "domains": card.domains,
                "now": _now(),
            },
        )

    async def upsert_author(self, name: str, paper_id: str, *, s2_id: str | None = None) -> None:
        key = s2_id or normalize_text(name)
        await self._store.execute(
            """
            MERGE (a:Author {workspace_id: $ws, key: $key})
            SET a.name = $name, a.s2_id = $s2
            WITH a
            MATCH (p:Paper {workspace_id: $ws, id: $pid})
            MERGE (a)-[:WROTE]->(p)
            """,
            {"ws": self._ws, "key": key, "name": name, "s2": s2_id, "pid": paper_id},
        )

    async def upsert_method(self, key: str, name: str, paper_id: str) -> None:
        await self._link_entity("Method", "USES_METHOD", key, name, paper_id)

    async def upsert_dataset(self, key: str, name: str, paper_id: str) -> None:
        await self._link_entity("Dataset", "USES_DATASET", key, name, paper_id)

    async def upsert_concept(self, key: str, name: str, paper_id: str) -> None:
        await self._link_entity("Concept", "ADDRESSES", key, name, paper_id)

    async def _link_entity(self, label: str, rel: str, key: str, name: str, paper_id: str) -> None:
        await self._store.execute(
            f"""
            MERGE (e:{label} {{workspace_id: $ws, key: $key}})
            SET e.name = $name
            WITH e
            MATCH (p:Paper {{workspace_id: $ws, id: $pid}})
            MERGE (p)-[:{rel}]->(e)
            """,
            {"ws": self._ws, "key": key, "name": name, "pid": paper_id},
        )

    async def upsert_claim(self, paper_id: str, text: str, evidence_location: str) -> str:
        claim_id = stable_id(self._ws, paper_id, text)
        await self._store.execute(
            """
            MERGE (c:Claim {workspace_id: $ws, id: $cid})
            SET c.text = $text, c.paper_id = $pid, c.evidence_location = $ev
            WITH c
            MATCH (p:Paper {workspace_id: $ws, id: $pid})
            MERGE (p)-[:MAKES_CLAIM]->(c)
            """,
            {
                "ws": self._ws,
                "cid": claim_id,
                "text": text,
                "pid": paper_id,
                "ev": evidence_location,
            },
        )
        return claim_id

    # ------------------------------------------------------------------ edges
    async def upsert_related_edge(self, edge: EdgeUpdate, valid_from: str | None = None) -> None:
        """MERGE a bi-temporal RELATED_TO edge with full weight anatomy."""
        await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws, id: $src})
            MATCH (b:Paper {workspace_id: $ws, id: $dst})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r.weight = $weight,
                r.weight_components = $components,
                r.computed_by_version = $version,
                r.valid_from = coalesce(r.valid_from, $valid_from),
                r.invalid_at = null,
                r.updated_at = $now
            """,
            {
                "ws": self._ws,
                "src": edge.source_id,
                "dst": edge.target_id,
                "weight": edge.weight,
                "components": json.dumps(edge.result.to_json()),
                "version": edge.computed_by_version,
                "valid_from": valid_from or _now(),
                "now": _now(),
            },
        )

    async def upsert_cites(
        self,
        source_id: str,
        target_id: str,
        *,
        context: str | None = None,
        intent: str | None = None,
    ) -> None:
        await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws, id: $src})
            MATCH (b:Paper {workspace_id: $ws, id: $dst})
            MERGE (a)-[r:CITES]->(b)
            SET r.context = $context, r.intent = $intent
            """,
            {
                "ws": self._ws,
                "src": source_id,
                "dst": target_id,
                "context": context,
                "intent": intent,
            },
        )

    async def upsert_claim_relation(
        self, source_id: str, target_id: str, relation: str, confidence: float
    ) -> None:
        """MERGE a SUPPORTS / CONTRADICTS / EXTENDS edge between two claims.

        The relation type is whitelisted to a known set to keep it injection-safe.
        """
        allowed = {"SUPPORTS", "CONTRADICTS", "EXTENDS"}
        if relation not in allowed:
            raise ValueError(f"invalid claim relation: {relation}")
        await self._store.execute(
            f"""
            MATCH (a:Claim {{workspace_id: $ws, id: $src}})
            MATCH (b:Claim {{workspace_id: $ws, id: $dst}})
            MERGE (a)-[r:{relation}]->(b)
            SET r.confidence = $confidence, r.computed_at = $now
            """,
            {
                "ws": self._ws,
                "src": source_id,
                "dst": target_id,
                "confidence": confidence,
                "now": _now(),
            },
        )

    async def set_superseded(self, superseded_id: str, superseding_id: str) -> None:
        await self._store.execute(
            """
            MATCH (old:Paper {workspace_id: $ws, id: $old})
            MATCH (new:Paper {workspace_id: $ws, id: $new})
            MERGE (old)-[r:SUPERSEDED_BY]->(new)
            SET r.created_at = coalesce(r.created_at, $now),
                old.superseded_by = $new
            """,
            {"ws": self._ws, "old": superseded_id, "new": superseding_id, "now": _now()},
        )

    async def invalidate_related_edge(self, source_id: str, target_id: str, reason: str) -> None:
        """Set ``invalid_at`` instead of deleting (bi-temporal supersession)."""
        await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws, id: $src})-[r:RELATED_TO]->(b:Paper {workspace_id: $ws, id: $dst})
            SET r.invalid_at = $now, r.invalid_reason = $reason
            """,
            {"ws": self._ws, "src": source_id, "dst": target_id, "now": _now(), "reason": reason},
        )

    async def invalidate_related_edges_of(self, paper_id: str, reason: str) -> None:
        """Invalidate every live RELATED_TO edge touching ``paper_id``, both
        directions (bi-temporal: superseded papers keep their history, hidden from
        default views by the ``invalid_at IS NULL`` read filters)."""
        await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws, id: $pid})-[r:RELATED_TO]-(b:Paper {workspace_id: $ws})
            WHERE r.invalid_at IS NULL
            SET r.invalid_at = $now, r.invalid_reason = $reason
            """,
            {"ws": self._ws, "pid": paper_id, "now": _now(), "reason": reason},
        )

    async def existing_related_weights(self, source_id: str) -> dict[str, float]:
        """Return current outgoing RELATED_TO weights for a paper (for auditing)."""
        rows = await self._store.execute(
            """
            MATCH (a:Paper {workspace_id: $ws, id: $src})-[r:RELATED_TO]->(b:Paper)
            WHERE r.invalid_at IS NULL
            RETURN b.id AS target, r.weight AS weight
            """,
            {"ws": self._ws, "src": source_id},
        )
        return {r["target"]: r["weight"] for r in rows}

    async def write_audit(self, entries: list[dict[str, Any]]) -> None:
        """Append an edge-weight-change trail as :EdgeAudit nodes on the graph store.

        MERGE (not CREATE) on (source, target, new_weight, reason) so a re-run of the
        linking stage (resume/retry) does not duplicate a change that was already
        recorded: a genuine weight change has a different new_weight and still adds a
        row; an identical replay is a no-op. `at` is stamped only on first insert.
        """
        for e in entries:
            await self._store.execute(
                """
                MERGE (x:EdgeAudit {workspace_id: $ws, source_id: $src, target_id: $dst,
                    new_weight: $new, reason: $reason})
                ON CREATE SET x.old_weight = $old, x.at = $at
                """,
                {"ws": self._ws, **e},
            )
