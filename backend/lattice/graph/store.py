"""Graph store abstraction.

All Cypher goes through :class:`GraphStore` so the writer/analytics layers are
testable without Neo4j (via :class:`FakeGraphStore`) and so Neo4j could be swapped
for an embedded engine (e.g. Kuzu) behind the same interface, per the no-vendor-
lock requirement.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from lattice.config import Neo4jSettings
from lattice.core.errors import CypherValidationError


class GraphStore(Protocol):
    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


class FakeGraphStore:
    """In-memory store for tests. Records every query and returns canned results.

    ``responses`` maps a substring of the query to a result list; the first match
    wins. Unmatched queries return an empty list. ``calls`` records (query, params)
    so writer/idempotency behavior can be asserted.
    """

    def __init__(self, responses: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((query, params or {}))
        for needle, result in self.responses.items():
            if needle in query:
                return result
        return []

    def queries_matching(self, pattern: str) -> list[str]:
        return [q for q, _ in self.calls if pattern in q]


class Neo4jGraphStore:  # pragma: no cover - requires a live Neo4j
    """Neo4j-backed store using the async driver."""

    def __init__(self, settings: Neo4jSettings) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            settings.uri, auth=(settings.user, settings.password)
        )
        self._database = settings.database

    async def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params or {})
            return [record.data() async for record in result]

    async def close(self) -> None:
        await self._driver.close()


# ---------------------------------------------------------------------------
# read-only Cypher validation for the agent's run_cypher tool
# ---------------------------------------------------------------------------
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|CALL\s*\{[^}]*(CREATE|MERGE|DELETE)|"
    r"LOAD\s+CSV|FOREACH|apoc\.\w+\.(create|merge|delete))\b",
    re.IGNORECASE,
)


def validate_read_only(query: str, max_limit: int = 1000) -> str:
    """Validate that a query is read-only and enforce a LIMIT.

    Used by the agent's ``run_cypher`` tool. Raises :class:`CypherValidationError`
    on any write clause. Appends a ``LIMIT`` if the query lacks one.
    """
    if ";" in query.rstrip().rstrip(";"):
        raise CypherValidationError("multiple statements are not allowed")
    if _WRITE_CLAUSES.search(query):
        raise CypherValidationError("only read-only queries are allowed")
    if not re.search(r"\bRETURN\b", query, re.IGNORECASE):
        raise CypherValidationError("query must contain a RETURN")
    if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        query = query.rstrip().rstrip(";") + f"\nLIMIT {max_limit}"
    return query
