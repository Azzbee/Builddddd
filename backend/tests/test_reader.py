from __future__ import annotations

import json

from lattice.graph.contradictions import ClaimRelation
from lattice.graph.reader import Neo4jGraphReader
from lattice.graph.store import FakeGraphStore


async def test_snapshot_maps_nodes_and_edges_and_normalizes_centrality() -> None:
    store = FakeGraphStore(
        {
            "RETURN p.id AS id": [
                {"id": "p1", "title": "A", "year": 2024, "community": 1, "centrality": 4.0, "needs_review": False},
                {"id": "p2", "title": "B", "year": 2023, "community": 1, "centrality": 2.0, "needs_review": True},
            ],
            "RELATED_TO]->(b:Paper": [
                {"source": "p1", "target": "p2", "weight": 0.83,
                 "components": json.dumps({"components": {"sem": 0.9, "meth": 0.5}})},
            ],
        }
    )
    snap = await Neo4jGraphReader(store).snapshot("ws")
    assert {n.id for n in snap.nodes} == {"p1", "p2"}
    p1 = next(n for n in snap.nodes if n.id == "p1")
    assert p1.centrality == 1.0  # 4.0 normalized by max 4.0
    assert snap.edges[0].weight == 0.83
    assert snap.edges[0].components == {"sem": 0.9, "meth": 0.5}


async def test_neighbors_filters_and_maps() -> None:
    store = FakeGraphStore(
        {"RELATED_TO]-(b:Paper": [{"source": "p1", "target": "p2", "weight": 0.7, "components": None}]}
    )
    edges = await Neo4jGraphReader(store).neighbors("ws", "p1", 0.5)
    assert edges[0].target == "p2" and edges[0].components == {}
    _, params = store.calls[-1]
    assert params["minw"] == 0.5


async def test_claim_relations_maps_relation_type() -> None:
    store = FakeGraphStore(
        {
            "type(r) IN": [
                {
                    "sid": "c1", "tid": "c2", "sp": "p1", "tp": "p2",
                    "relation": "CONTRADICTS", "confidence": 0.8,
                    "stext": "improves", "ttext": "no improvement",
                    "sev": "T1", "tev": "T2",
                }
            ]
        }
    )
    edges = await Neo4jGraphReader(store).claim_relations("ws", None)
    assert edges[0].relation == ClaimRelation.CONTRADICTS
    assert edges[0].source_paper == "p1" and edges[0].confidence == 0.8


async def test_lineage_filters_to_method_members_and_orders() -> None:
    store = FakeGraphStore(
        {
            "USES_METHOD]->(m:Method)": [
                {"id": "p1", "title": "Seminal", "year": 2016},
                {"id": "p2", "title": "Newer", "year": 2020},
            ],
            "[:CITES]->(b:Paper": [{"citing": "p2", "cited": "p1"}],
        }
    )
    lin = await Neo4jGraphReader(store).lineage("ws", "lstm")
    assert [n.paper_id for n in lin.nodes] == ["p1", "p2"]  # sorted by year
    assert lin.edges[0].source == "p1" and lin.edges[0].target == "p2"  # older -> newer
