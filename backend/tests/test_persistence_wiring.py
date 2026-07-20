"""build_container wires persistent backends when enabled (offline, with fakes)."""

from __future__ import annotations

import pytest
from lattice.api import deps
from lattice.api.deps import build_container
from lattice.config import Settings
from lattice.db.ingest_artifacts import PgIngestArtifactStore
from lattice.db.pg_stores import PgCardStore, PgJobStore
from lattice.db.vector import PgVectorStore
from lattice.graph.reader import Neo4jGraphReader
from lattice.graph.store import FakeGraphStore


@pytest.fixture(autouse=True)
def _reset_persist():
    yield
    deps._persist_pool = None
    deps._persist_graph = None
    deps._persist_redis = None


def test_in_memory_by_default() -> None:
    c = build_container(Settings())
    assert c.ingestion.reader is None
    assert type(c.cards).__name__ == "InMemoryCardStore"


def test_persistent_branch_wires_pg_and_neo4j() -> None:
    deps._persist_pool = object()  # stand-in pool; not used without a real query
    deps._persist_graph = FakeGraphStore()
    c = build_container(Settings(persistent=True))
    assert isinstance(c.vectors, PgVectorStore)
    assert isinstance(c.cards, PgCardStore)
    assert isinstance(c.jobs, PgJobStore)
    assert isinstance(c.artifacts, PgIngestArtifactStore)
    assert isinstance(c.ingestion.reader, Neo4jGraphReader)


async def test_init_persistence_noop_when_disabled() -> None:
    await deps.init_persistence(Settings(persistent=False))
    assert deps._persist_pool is None


class FakeConnection:
    async def fetchval(self, query: str) -> int:
        assert query == "SELECT 1"
        return 1


class FakeAcquireContext:
    async def __aenter__(self) -> FakeConnection:
        return FakeConnection()

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePool:
    def acquire(self) -> FakeAcquireContext:
        return FakeAcquireContext()


class FakeRedis:
    async def ping(self) -> bool:
        return True


async def test_persistence_health_probes_all_backends() -> None:
    deps._persist_pool = FakePool()
    deps._persist_graph = FakeGraphStore({"RETURN 1": [{"ok": 1}]})
    deps._persist_redis = FakeRedis()

    checks = await deps.persistence_health(Settings(persistent=True))

    assert checks == {"postgres": True, "neo4j": True, "redis": True}


async def test_persistence_health_marks_missing_backend_unready() -> None:
    deps._persist_pool = FakePool()
    deps._persist_graph = FakeGraphStore({"RETURN 1": [{"ok": 1}]})

    checks = await deps.persistence_health(Settings(persistent=True))

    assert checks == {"postgres": True, "neo4j": True, "redis": False}
