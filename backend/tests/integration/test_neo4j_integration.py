"""Live Neo4j integration tests.

Gated on ``LATTICE_TEST_NEO4J_URI``. Exercises the real Neo4jGraphStore: schema
constraints, idempotent MERGE writes, bi-temporal RELATED_TO edges, claim
relations, and read-back. Analytics (GDS) is tested only if the plugin is present.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration

URI = os.environ.get("LATTICE_TEST_NEO4J_URI")
if not URI:
    pytest.skip("LATTICE_TEST_NEO4J_URI not set", allow_module_level=True)

from lattice.config import Neo4jSettings, SimilarityWeights  # noqa: E402
from lattice.extraction.schemas import Methodology, PaperCard, PaperType  # noqa: E402
from lattice.graph.evolution import compute_related_edges  # noqa: E402
from lattice.graph.schema import apply_schema  # noqa: E402
from lattice.graph.similarity import CosineCalibrator, PaperFeatures  # noqa: E402
from lattice.graph.store import Neo4jGraphStore  # noqa: E402
from lattice.graph.writer import GraphWriter  # noqa: E402


def _settings() -> Neo4jSettings:
    return Neo4jSettings(
        uri=URI,
        user=os.environ.get("LATTICE_TEST_NEO4J_USER", "neo4j"),
        password=os.environ.get("LATTICE_TEST_NEO4J_PASSWORD", "lattice-dev-password"),
    )


@pytest.fixture
async def store():  # type: ignore[no-untyped-def]
    s = Neo4jGraphStore(_settings())
    await s.execute("MATCH (n) WHERE n.workspace_id = 'itest' DETACH DELETE n")
    await apply_schema(s)
    yield s
    await s.execute("MATCH (n) WHERE n.workspace_id = 'itest' DETACH DELETE n")
    await s.close()


def _card(pid: str, title: str) -> PaperCard:
    return PaperCard(
        paper_id=pid, title=title, year=2024,
        methodology=Methodology(approach_summary="LSTM"), paper_type=PaperType.EMPIRICAL,
        domains=["commodity markets"], methods_taxonomy=["LSTM"],
    )


async def test_paper_upsert_idempotent(store) -> None:  # type: ignore[no-untyped-def]
    writer = GraphWriter(store, "itest")
    await writer.upsert_paper(_card("p1", "Copper LSTM"))
    await writer.upsert_paper(_card("p1", "Copper LSTM"))  # second write must not duplicate
    rows = await store.execute(
        "MATCH (p:Paper {workspace_id:'itest', id:'p1'}) RETURN count(p) AS n"
    )
    assert rows[0]["n"] == 1


async def test_related_edge_roundtrip(store) -> None:  # type: ignore[no-untyped-def]
    writer = GraphWriter(store, "itest")
    await writer.upsert_paper(_card("p1", "A"))
    await writer.upsert_paper(_card("p2", "B"))
    feats_a = PaperFeatures("p1", specter=np.array([1.0, 0.0]), methods={"lstm"})
    edges = compute_related_edges(
        feats_a, [PaperFeatures("p2", specter=np.array([1.0, 0.0]), methods={"lstm"})],
        SimilarityWeights(), CosineCalibrator(),
    )
    assert edges
    await writer.upsert_related_edge(edges[0])
    weights = await writer.existing_related_weights("p1")
    assert "p2" in weights and weights["p2"] > 0


async def test_claim_relation_and_entities(store) -> None:  # type: ignore[no-untyped-def]
    writer = GraphWriter(store, "itest")
    await writer.upsert_paper(_card("p1", "A"))
    await writer.upsert_method("k-lstm", "LSTM", "p1")
    c1 = await writer.upsert_claim("p1", "LSTM beats ARIMA", "Table 1")
    await writer.upsert_paper(_card("p2", "B"))
    c2 = await writer.upsert_claim("p2", "LSTM shows no improvement", "Table 2")
    await writer.upsert_claim_relation(c1, c2, "CONTRADICTS", 0.8)
    rows = await store.execute(
        "MATCH (:Claim {workspace_id:'itest'})-[r:CONTRADICTS]->(:Claim) RETURN count(r) AS n"
    )
    assert rows[0]["n"] == 1
    methods = await store.execute(
        "MATCH (:Paper {workspace_id:'itest', id:'p1'})-[:USES_METHOD]->(m:Method) RETURN m.name AS name"
    )
    assert methods[0]["name"] == "LSTM"
