"""Tests for the gap -> research-proposal generator (pure builder)."""

from __future__ import annotations

from lattice.landscape.proposal import (
    BLIND_SPOT_MIN_GLOBAL,
    FacetPaper,
    MomentumLite,
    build_proposal,
)


def _papers() -> list[FacetPaper]:
    return [
        FacetPaper(
            "p1",
            "LSTM on copper",
            2019,
            {"lstm"},
            {"lme copper"},
            "LSTM beats ARIMA",
            ["single commodity"],
        ),
        FacetPaper(
            "p2",
            "LSTM on aluminium",
            2020,
            {"lstm"},
            {"lme aluminium"},
            "LSTM generalizes",
            ["no transaction costs"],
        ),
        FacetPaper(
            "p3", "VAR on gold", 2018, {"var"}, {"comex gold"}, "VAR underperforms", ["linear only"]
        ),
    ]


def test_proposal_for_true_gap_composes_building_blocks() -> None:
    # Gap: lstm x comex gold. lstm has a track record; comex gold studied with var.
    p = build_proposal(
        "method",
        "dataset",
        "lstm",
        "comex gold",
        _papers(),
        now_year=2024,
        global_count=3,
        demand=0.6,
    )
    assert p.state == "gap"
    assert "lightly explored" in p.novelty.lower()
    # Building blocks: lstm's two applications + the gold paper.
    assert {e.paper_id for e in p.row_track_record} == {"p1", "p2"}
    assert {e.paper_id for e in p.col_track_record} == {"p3"}
    # Method to borrow is the most recent lstm application.
    assert p.method_to_borrow is not None and p.method_to_borrow.paper_id == "p2"
    # Baselines to beat are the methods already used on the dataset.
    assert "var" in p.recommended_baselines
    # Risks aggregate the flanking limitations.
    assert any("single commodity" in r for r in p.risks)
    assert "# Research proposal" in p.markdown()
    assert 0.0 <= p.confidence <= 1.0


def test_blind_spot_framing_when_world_has_many() -> None:
    p = build_proposal(
        "method",
        "dataset",
        "lstm",
        "comex gold",
        _papers(),
        now_year=2024,
        global_count=BLIND_SPOT_MIN_GLOBAL + 10,
        demand=0.3,
    )
    assert p.state == "blind_spot"
    assert "blind spot" in p.novelty.lower()
    assert any("reinventing" in r.lower() for r in p.risks)


def test_greenfield_when_no_external_signal() -> None:
    p = build_proposal(
        "method",
        "dataset",
        "lstm",
        "comex gold",
        _papers(),
        now_year=2024,
        global_count=0,
        demand=0.1,
    )
    assert p.state == "gap"
    assert "greenfield" in p.novelty.lower()
    assert any("infeasibility" in r.lower() for r in p.risks)


def test_occupied_cell_reports_prior_art_not_a_gap() -> None:
    papers = [
        *_papers(),
        FacetPaper("p4", "LSTM on gold", 2023, {"lstm"}, {"comex gold"}, "works", []),
    ]
    p = build_proposal("method", "dataset", "lstm", "comex gold", papers, now_year=2024)
    assert p.state == "occupied"
    assert "already" in p.novelty.lower()
    assert p.method_to_borrow is None  # nothing to "borrow"; it is done


def test_missing_track_record_lowers_confidence_and_flags_risk() -> None:
    p = build_proposal(
        "method",
        "dataset",
        "transformer",
        "eia energy",
        _papers(),
        now_year=2024,
        global_count=0,
        demand=0.0,
    )
    # Neither side has corpus precedent.
    assert not p.row_track_record and not p.col_track_record
    assert any("no corpus track record" in r.lower() for r in p.risks)
    assert p.confidence == 0.0


def test_why_now_uses_momentum() -> None:
    hot = MomentumLite(maturity="accelerating", composite=0.8, burst=2.0)
    p = build_proposal(
        "method",
        "dataset",
        "lstm",
        "comex gold",
        _papers(),
        now_year=2024,
        demand=0.5,
        row_momentum=hot,
    )
    assert "accelerating" in p.why_now.lower()
    assert "demand" in p.why_now.lower()


def test_open_problems_included() -> None:
    p = build_proposal(
        "method",
        "dataset",
        "lstm",
        "comex gold",
        _papers(),
        now_year=2024,
        open_problems=["extend LSTM to gold markets"],
    )
    assert p.open_problems == ["extend LSTM to gold markets"]
    assert "Open problems" in p.markdown()
