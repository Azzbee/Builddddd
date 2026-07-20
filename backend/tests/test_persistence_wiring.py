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
