from __future__ import annotations

import json

import numpy as np
import pytest
from lattice.config import SimilarityWeights
from lattice.core.errors import CypherValidationError
from lattice.extraction.schemas import Methodology, PaperCard, PaperType
from lattice.graph.analytics import recompute_analytics, top_papers_by_pagerank
from lattice.graph.entity_resolution import EntityResolver
from lattice.graph.evolution import (
    CitationSignal,
    compute_related_edges,
    detect_supersession,
    diff_audit,
)
from lattice.graph.schema import CONSTRAINTS, apply_schema
from lattice.graph.similarity import CosineCalibrator, PaperFeatures
from lattice.graph.store import FakeGraphStore, validate_read_only
from lattice.graph.writer import GraphWriter


# --------------------------------------------------------------------------- store / cypher guard
def test_validate_read_only_allows_select() -> None:
    q = validate_read_only("MATCH (p:Paper) RETURN p.title")
    assert "LIMIT" in q


@pytest.mark.parametrize(
    "bad",
    [
        "MATCH (p) DELETE p",
        "CREATE (p:Paper) RETURN p",
        "MATCH (p) SET p.x = 1 RETURN p",
        "MATCH (p) RETURN p; MATCH (q) RETURN q",
        "MATCH (p) WHERE p.x=1",  # no RETURN
    ],
)
def test_validate_read_only_rejects_writes(bad: str) -> None:
    with pytest.raises(CypherValidationError):
        validate_read_only(bad)


def test_validate_read_only_keeps_existing_limit() -> None:
    q = validate_read_only("MATCH (p) RETURN p LIMIT 5")
    assert q.count("LIMIT") == 1


async def test_apply_schema_runs_all_constraints() -> None:
    store = FakeGraphStore()
    await apply_schema(store)
    assert len(store.calls) == len(CONSTRAINTS) + 3  # + indexes
    assert any("CONSTRAINT paper_id" in q for q, _ in store.calls)


# --------------------------------------------------------------------------- entity resolution
def test_entity_resolver_exact_and_new() -> None:
    r = EntityResolver("method")
    res1 = r.resolve("LSTM")
    assert not res1.matched and res1.method == "new"
    res2 = r.resolve("lstm")  # normalized exact
    assert res2.matched and res2.method == "exact" and res2.key == res1.key


def test_entity_resolver_fuzzy_merge() -> None:
    r = EntityResolver("method")
    r.register("Random Forest Classifier")
    res = r.resolve("Random Forests Classifier")  # ~0.98 -> fuzzy auto-match
    assert res.matched and res.method == "fuzzy"


def test_entity_resolver_embedding_tiebreak() -> None:
    # "vector autoregression model" vs "vector autoregressive models" ~0.909
    # lands in the ambiguous band; near-identical embeddings resolve it.
    r = EntityResolver("method")
    r.register("vector autoregression model", embedding=[1.0, 0.0, 0.0])
    res = r.resolve("vector autoregressive models", embedding=[0.99, 0.01, 0.0])
    assert res.matched and res.method == "embedding"


def test_entity_resolver_embedding_rejects_distant() -> None:
    r = EntityResolver("method")
    r.register("vector autoregression model", embedding=[1.0, 0.0])
    # Ambiguous lexically but embedding-distant and no tiebreaker -> new.
    res = r.resolve("vector autoregressive models", embedding=[0.0, 1.0])
    assert not res.matched and res.method == "new"


def test_entity_resolver_llm_tiebreak() -> None:
    calls: list[tuple[str, str]] = []

    def tb(a: str, b: str) -> bool:
        calls.append((a, b))
        return True

    r = EntityResolver("method", tiebreaker=tb)
    r.register("vector autoregression model")
    res = r.resolve("vector autoregressive models")  # ambiguous band, no embeddings
    assert res.method == "llm" and res.matched and calls


# --------------------------------------------------------------------------- evolution
def _feat(pid: str, vec: list[float], **kw) -> PaperFeatures:
    return PaperFeatures(pid, specter=np.array(vec, dtype=float), **kw)


def test_compute_related_edges_thresholds_and_cap() -> None:
    w = SimilarityWeights(knn_cap=2, tau=0.35)
    cal = CosineCalibrator(lo=0.0, hi=1.0, fitted=True)
    new = _feat("new", [1.0, 0.0], methods={"lstm"})
    cands = [
        _feat("a", [1.0, 0.0], methods={"lstm"}),  # very similar
        _feat("b", [0.9, 0.1], methods={"lstm"}),  # similar
        _feat("c", [0.0, 1.0], methods={"garch"}),  # orthogonal -> below tau
    ]
    edges = compute_related_edges(new, cands, w, cal)
    ids = [e.target_id for e in edges]
    assert "c" not in ids  # below tau
    assert len(edges) <= 2  # knn cap
    assert edges == sorted(edges, key=lambda e: e.weight, reverse=True)


def test_compute_related_edges_skips_self() -> None:
    w = SimilarityWeights()
    new = _feat("x", [1.0, 0.0])
    edges = compute_related_edges(new, [_feat("x", [1.0, 0.0])], w, CosineCalibrator())
    assert edges == []


def test_compute_related_edges_uses_citation_signal() -> None:
    w = SimilarityWeights(tau=0.35)
    cal = CosineCalibrator(lo=0.0, hi=1.0, fitted=True)
    new = _feat("new", [0.2, 1.0])
    cand = _feat("cited", [1.0, 0.2])
    cites = {"cited": CitationSignal(direct_citation=True)}
    edges = compute_related_edges(new, [cand], w, cal, citations=cites)
    assert edges and "cit" in edges[0].result.available
    assert edges[0].result.components["cit"] == 1.0


def test_diff_audit_records_changes_only() -> None:
    w = SimilarityWeights()
    cal = CosineCalibrator(lo=0.0, hi=1.0, fitted=True)
    new = _feat("new", [1.0, 0.0], methods={"lstm"})
    edges = compute_related_edges(new, [_feat("a", [1.0, 0.0], methods={"lstm"})], w, cal)
    assert edges
    # No existing edge -> audited as new.
    fresh = diff_audit(edges, {})
    assert len(fresh) == 1 and fresh[0].old_weight is None
    # Same weight already present -> not audited.
    same = diff_audit(edges, {edges[0].target_id: edges[0].weight})
    assert same == []


def test_detect_supersession() -> None:
    # new is published (doi, not preprint); candidate is arXiv preprint.
    assert detect_supersession("new", True, False, ("old", False, True)) == ("old", "new")
    # new is preprint; candidate published.
    assert detect_supersession("new", False, True, ("old", True, False)) == ("new", "old")
    # both published -> no supersession.
    assert detect_supersession("new", True, False, ("old", True, False)) is None


# --------------------------------------------------------------------------- writer idempotency
def _card() -> PaperCard:
    return PaperCard(
        paper_id="p1",
        title="Copper forecasting",
        year=2024,
        doi="10.1/x",
        methodology=Methodology(approach_summary="LSTM"),
        paper_type=PaperType.EMPIRICAL,
        domains=["commodity markets"],
    )


async def test_writer_uses_merge_for_idempotency() -> None:
    store = FakeGraphStore()
    writer = GraphWriter(store, workspace_id="ws1")
    await writer.upsert_paper(_card())
    await writer.upsert_author("Jane Doe", "p1", s2_id="a1")
    await writer.upsert_method("k-lstm", "LSTM", "p1")
    # Every node/edge write uses MERGE (idempotent), never bare CREATE.
    for q, params in store.calls:
        assert "MERGE" in q
        assert params.get("ws") == "ws1"
    assert any("USES_METHOD" in q for q, _ in store.calls)


async def test_writer_audit_uses_merge_not_create() -> None:
    # The audit trail must be idempotent on a linking-stage replay: MERGE on the
    # change identity, never a bare CREATE (which duplicated rows on retry).
    store = FakeGraphStore()
    writer = GraphWriter(store, "ws1")
    entry = {
        "src": "p1",
        "dst": "p2",
        "old": 0.4,
        "new": 0.6,
        "reason": "ingest",
        "at": "2024-01-01",
    }
    await writer.write_audit([entry])
    q, _ = store.calls[-1]
    # MERGE on the change identity; only `ON CREATE SET` may mention CREATE, never a
    # bare `CREATE (` node write (which duplicated on replay).
    assert "MERGE (" in q and "CREATE (" not in q
    assert "new_weight: $new" in q and "reason: $reason" in q


async def test_writer_related_edge_serializes_components() -> None:
    w = SimilarityWeights()
    cal = CosineCalibrator(lo=0.0, hi=1.0, fitted=True)
    new = _feat("p1", [1.0, 0.0], methods={"lstm"})
    edges = compute_related_edges(new, [_feat("p2", [1.0, 0.0], methods={"lstm"})], w, cal)
    store = FakeGraphStore()
    writer = GraphWriter(store, "ws1")
    await writer.upsert_related_edge(edges[0])
    q, params = store.calls[-1]
    assert "RELATED_TO" in q and "invalid_at = null" in q
    components = json.loads(params["components"])
    assert "components" in components and "weight" in components


async def test_writer_invalidate_sets_invalid_at() -> None:
    store = FakeGraphStore()
    writer = GraphWriter(store, "ws1")
    await writer.invalidate_related_edge("p1", "p2", reason="superseded")
    q, params = store.calls[-1]
    assert "invalid_at = $now" in q and params["reason"] == "superseded"


async def test_writer_existing_weights_reads() -> None:
    store = FakeGraphStore({"RETURN b.id AS target": [{"target": "p2", "weight": 0.5}]})
    writer = GraphWriter(store, "ws1")
    weights = await writer.existing_related_weights("p1")
    assert weights == {"p2": 0.5}


# --------------------------------------------------------------------------- analytics
async def test_recompute_analytics_runs_gds_algorithms() -> None:
    store = FakeGraphStore({"count(p) AS papers": [{"papers": 20, "communities": 4}]})
    summary = await recompute_analytics(store, "ws1")
    assert summary == {"papers": 20, "communities": 4}
    joined = "\n".join(q for q, _ in store.calls)
    assert "gds.pageRank.write" in joined
    assert "gds.louvain.write" in joined
    assert "gds.betweenness.write" in joined


async def test_top_papers_by_pagerank_parses_rows() -> None:
    store = FakeGraphStore(
        {"ORDER BY p.pagerank DESC": [{"id": "p1", "title": "T", "pagerank": 0.9, "community": 1}]}
    )
    rows = await top_papers_by_pagerank(store, "ws1", limit=5)
    assert rows[0]["id"] == "p1"
