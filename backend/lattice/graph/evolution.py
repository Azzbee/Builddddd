"""The incremental update protocol: the 'graphe evolutif'.

On ingesting paper p, we never recompute all pairs. Instead:

1. Candidate generation (top-k by SPECTER2 ANN, done upstream in pgvector).
2. Compute the full composite similarity only against those candidates.
3. Materialize RELATED_TO edges above tau, capped at k-NN per paper.
4. Every weight change is recorded in an audit trail; superseded/contradicted
   edges get ``invalid_at`` set rather than being deleted (bi-temporal).

This module is pure: it computes the edge set and audit entries. Persistence is
done by the writer/linker that consumes these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from lattice.config import SimilarityWeights
from lattice.graph.similarity import (
    CosineCalibrator,
    PaperFeatures,
    WeightResult,
    composite_weight,
)


@dataclass
class CitationSignal:
    co_citation: float | None = None
    direct_citation: bool = False


@dataclass
class EdgeUpdate:
    source_id: str
    target_id: str
    result: WeightResult
    computed_by_version: str

    @property
    def weight(self) -> float:
        return self.result.weight


@dataclass
class AuditEntry:
    source_id: str
    target_id: str
    old_weight: float | None
    new_weight: float
    reason: str
    components: dict[str, float] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


def compute_related_edges(
    new: PaperFeatures,
    candidates: list[PaperFeatures],
    weights: SimilarityWeights,
    calibrator: CosineCalibrator | None = None,
    *,
    citations: dict[str, CitationSignal] | None = None,
    version: str = "sim_v1",
    now: datetime | None = None,
) -> list[EdgeUpdate]:
    """Compute the RELATED_TO edges for a newly ingested paper.

    Edges below ``tau`` are dropped; the rest are sorted by weight and capped at
    ``knn_cap``. ``citations`` maps a candidate paper_id to its citation signal.
    """
    citations = citations or {}
    edges: list[EdgeUpdate] = []
    for cand in candidates:
        if cand.paper_id == new.paper_id:
            continue
        sig = citations.get(cand.paper_id, CitationSignal())
        result = composite_weight(
            new,
            cand,
            weights,
            calibrator,
            co_citation=sig.co_citation,
            direct_citation=sig.direct_citation,
            now=now,
        )
        if result.weight >= weights.tau:
            edges.append(EdgeUpdate(new.paper_id, cand.paper_id, result, version))

    edges.sort(key=lambda e: e.weight, reverse=True)
    return edges[: weights.knn_cap]


def diff_audit(
    new_edges: list[EdgeUpdate],
    existing_weights: dict[str, float],
    reason: str = "ingest",
) -> list[AuditEntry]:
    """Produce audit entries for changed edges (nothing silently overwritten).

    ``existing_weights`` maps target_id -> current weight for the source paper.
    """
    entries: list[AuditEntry] = []
    for edge in new_edges:
        old = existing_weights.get(edge.target_id)
        if old is None or abs(old - edge.weight) > 1e-9:
            entries.append(
                AuditEntry(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    old_weight=old,
                    new_weight=edge.weight,
                    reason=reason,
                    components=edge.result.components,
                )
            )
    return entries


def detect_supersession(
    new_id: str,
    new_has_doi: bool,
    new_is_preprint: bool,
    candidate: tuple[str, bool, bool],
) -> tuple[str, str] | None:
    """Return (superseded_id, superseding_id) if one paper supersedes the other.

    Heuristic for preprint->published: the version with a DOI supersedes the
    preprint-only (arXiv) version. ``candidate`` is (id, has_doi, is_preprint).
    Caller guarantees the two are the same logical paper (matched title+authors).
    """
    cand_id, cand_has_doi, cand_is_preprint = candidate
    if new_has_doi and not new_is_preprint and cand_is_preprint and not cand_has_doi:
        return (cand_id, new_id)  # candidate (preprint) superseded by new (published)
    if cand_has_doi and not cand_is_preprint and new_is_preprint and not new_has_doi:
        return (new_id, cand_id)  # new (preprint) superseded by candidate (published)
    return None
