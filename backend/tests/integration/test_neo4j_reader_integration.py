"""Live Neo4j reader integration: write via GraphWriter, read via Neo4jGraphReader."""

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
from lattice.graph.reader import Neo4jGraphReader  # noqa: E402
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


def _card(pid: str, title: str, method: str) -> PaperCard:
    return PaperCard(
        paper_id=pid, title=title, year=2020 if pid == "p1" else 2023,
        methodology=Methodology(approach_summary=method), paper_type=PaperType.EMPIRICAL,
        domains=["commodity markets"], methods_taxonomy=[method],
    )


@pytest.fixture
async def store():  # type: ignore[no-untyped-def]
    s = Neo4jGraphStore(_settings())
    await s.execute("MATCH (n) WHERE n.workspace_id = 'rtest' DETACH DELETE n")
    await apply_schema(s)
    yield s
    await s.execute("MATCH (n) WHERE n.workspace_id = 'rtest' DETACH DELETE n")
    await s.close()


async def test_reader_snapshot_and_lineage_and_relations(store) -> None:  # type: ignore[no-untyped-def]
    writer = GraphWriter(store, "rtest")
    await writer.upsert_paper(_card("p1", "Seminal LSTM", "LSTM"))
    await writer.upsert_paper(_card("p2", "LSTM redux", "LSTM"))
    await writer.upsert_method("k-lstm", "LSTM", "p1")
    await writer.upsert_method("k-lstm", "LSTM", "p2")
    await writer.upsert_cites("p2", "p1")

    feats = PaperFeatures("p1", specter=np.array([1.0, 0.0]), methods={"lstm"})
    edges = compute_related_edges(
        feats, [PaperFeatures("p2", specter=np.array([1.0, 0.0]), methods={"lstm"})],
        SimilarityWeights(), CosineCalibrator(),
    )
    await writer.upsert_related_edge(edges[0])

    c1 = await writer.upsert_claim("p1", "LSTM improves accuracy", "T1")
    c2 = await writer.upsert_claim("p2", "LSTM shows no improvement", "T2")
    await writer.upsert_claim_relation(c1, c2, "CONTRADICTS", 0.9)

    reader = Neo4jGraphReader(store)

    snap = await reader.snapshot("rtest")
    assert {n.id for n in snap.nodes} == {"p1", "p2"}
    assert any(e.source in {"p1", "p2"} and e.target in {"p1", "p2"} for e in snap.edges)

    # Time-travel: as of 2021 only p1 (2020) exists, and the p1<->p2 edge is gone.
    snap_2021 = await reader.snapshot("rtest", as_of_year=2021)
    assert {n.id for n in snap_2021.nodes} == {"p1"}
    assert snap_2021.edges == []

    neighbors = await reader.neighbors("rtest", "p1", 0.0)
    assert any(e.target == "p2" or e.source == "p2" for e in neighbors)

    rels = await reader.claim_relations("rtest", "CONTRADICTS")
    assert rels and rels[0].relation.value == "CONTRADICTS"

    lin = await reader.lineage("rtest", "LSTM")
    assert [n.paper_id for n in lin.nodes] == ["p1", "p2"]  # ordered by year
    assert lin.edges and lin.edges[0].source == "p1" and lin.edges[0].target == "p2"
