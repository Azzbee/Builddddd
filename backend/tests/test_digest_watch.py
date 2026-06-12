from __future__ import annotations

from lattice.digest.weekly import (
    Contradiction,
    DigestInput,
    NewPaper,
    WeightChange,
    build_digest,
    render_markdown,
)
from lattice.embeddings.base import HashingEmbedder
from lattice.enrichment.models import ArxivResult
from lattice.landscape.momentum import momentum_score
from lattice.watch import score_candidates


def test_build_digest_thresholds_and_ranks() -> None:
    data = DigestInput(
        period_label="2026-W24",
        new_papers=[NewPaper("p1", "Copper LSTM", 2026)],
        new_edges=12,
        weight_changes=[
            WeightChange("p1", "p2", 0.5, 0.55),  # small, dropped
            WeightChange("p1", "p3", 0.2, 0.6),  # large, kept
        ],
        contradictions=[Contradiction("A improves Y", "p1", "no effect", "p2")],
        movers=[
            momentum_score("transformers", {2024: 1, 2025: 5, 2026: 20}, community_convergence=4),
            momentum_score("garch", {2024: 5, 2025: 5, 2026: 5}),
        ],
    )
    report = build_digest(data, weight_change_min=0.1)
    assert report.summary["new_papers"] == 1
    assert len(report.notable_weight_changes) == 1  # small change filtered
    assert report.notable_weight_changes[0].target_id == "p3"
    # Movers ranked by composite; transformers should lead.
    assert report.movers[0].concept == "transformers"


def test_render_markdown_no_em_dash() -> None:
    data = DigestInput(
        period_label="2026-W24",
        new_papers=[NewPaper("p1", "Copper LSTM", 2026)],
        new_edges=3,
        movers=[momentum_score("transformers", {2024: 1, 2025: 5, 2026: 20})],
    )
    md = render_markdown(build_digest(data))
    assert "# Lattice digest: 2026-W24" in md
    assert "## New papers" in md
    assert "—" not in md  # no em dashes, per house style


def test_score_candidates_filters_by_similarity() -> None:
    embedder = HashingEmbedder(dim=256)
    # Build corpus vectors from two distinct topics.
    corpus_texts = {
        "p1": "lstm attention copper price forecasting commodity markets",
        "p2": "reinforcement learning robot locomotion control",
    }
    corpus = [(pid, embedder.embed([t])[0]) for pid, t in corpus_texts.items()]

    candidates = [
        ArxivResult(
            arxiv_id="2406.1",
            title="Transformer copper price forecasting",
            summary="forecasting commodity prices with attention and lstm models",
        ),
        ArxivResult(
            arxiv_id="2406.2",
            title="Quantum chromodynamics lattice simulation",
            summary="particle physics lattice gauge theory simulation",
        ),
    ]
    scored = score_candidates(candidates, corpus, embedder=embedder, floor=0.1)
    ids = [s.candidate.arxiv_id for s in scored]
    assert "2406.1" in ids
    # The on-topic candidate should be the nearest to the forecasting paper.
    top = scored[0]
    assert top.candidate.arxiv_id == "2406.1"
    assert top.nearest_paper_id == "p1"


def test_score_candidates_empty_inputs() -> None:
    assert score_candidates([], []) == []
