from __future__ import annotations

from lattice.landscape.matrix import (
    CellState,
    PaperFacets,
    build_gap_matrix,
    top_gaps,
)
from lattice.landscape.momentum import (
    MaturityStage,
    acceleration,
    burst_score,
    maturity_stage,
    momentum_score,
    velocity,
)
from lattice.landscape.quadrants import (
    OpenProblemMention,
    SupportingPaper,
    cluster_claims,
    cluster_open_problems,
    consolidate_known_knowns,
    cross_community_transfer,
)


# --------------------------------------------------------------------------- momentum
def test_velocity_and_acceleration() -> None:
    counts = {2020: 1, 2021: 2, 2022: 4, 2023: 8}
    assert velocity(counts) > 0
    assert acceleration(counts) > 0  # growth is speeding up


def test_burst_score_detects_spike() -> None:
    flat = {2020: 5, 2021: 5, 2022: 5}
    spiky = {2020: 1, 2021: 1, 2022: 20}
    assert burst_score(flat) == 0.0
    assert burst_score(spiky) > 1.0


def test_maturity_stages() -> None:
    accelerating = {2020: 1, 2021: 2, 2022: 10}
    assert maturity_stage(accelerating) == MaturityStage.ACCELERATING
    saturating = {2018: 50, 2019: 60, 2020: 61, 2021: 61}
    assert maturity_stage(saturating) in (MaturityStage.SATURATING, MaturityStage.EMERGING)
    assert maturity_stage({2020: 1}) == MaturityStage.UNKNOWN


def test_momentum_score_composite_visible() -> None:
    counts = {2020: 1, 2021: 3, 2022: 9, 2023: 20}
    score = momentum_score("transformers", counts, author_influx=0.6, community_convergence=4)
    assert 0.0 <= score.composite <= 1.0
    js = score.to_json()
    assert js["concept"] == "transformers"
    assert "velocity" in js and "acceleration" in js and "maturity" in js


# --------------------------------------------------------------------------- gap matrix
def _papers() -> list[PaperFacets]:
    return [
        PaperFacets("p1", 2024, methods={"LSTM"}, datasets={"LME"}),
        PaperFacets("p2", 2024, methods={"LSTM"}, datasets={"LME"}),
        PaperFacets("p3", 2015, methods={"VAR"}, datasets={"LME"}),  # stale
        PaperFacets("p4", 2024, methods={"VAR"}, datasets={"COMEX"}),
    ]


def test_gap_matrix_classifies_cells() -> None:
    cells = build_gap_matrix(
        _papers(), "method", "dataset", now_year=2026,
        global_count_fn=lambda r, c: 100 if (r, c) == ("LSTM", "COMEX") else 0,
    )
    by_key = {(c.row, c.col): c for c in cells}
    assert by_key[("LSTM", "LME")].state == CellState.ACTIVE
    assert by_key[("VAR", "LME")].state == CellState.STALE
    # Empty corpus cell but high global count -> blind spot (reading gap).
    assert by_key[("LSTM", "COMEX")].state == CellState.BLIND_SPOT
    # Empty everywhere (no global presence) -> empty.
    assert by_key[("VAR", "COMEX")].state in (CellState.ACTIVE, CellState.EMPTY)


def test_gap_matrix_contested_via_contradiction() -> None:
    papers = [
        PaperFacets("a", 2024, methods={"LSTM"}, datasets={"LME"}),
        PaperFacets("b", 2024, methods={"LSTM"}, datasets={"LME"}),
    ]
    cells = build_gap_matrix(
        papers, "method", "dataset", now_year=2026,
        contradiction_pairs={frozenset({"a", "b"})},
    )
    assert cells[0].state == CellState.CONTESTED


def test_gap_matrix_structurally_empty_excluded_from_gaps() -> None:
    cells = build_gap_matrix(
        _papers(), "method", "dataset", now_year=2026,
        feasibility_fn=lambda r, c: 0.0,  # everything nonsensical
        global_count_fn=lambda r, c: 50,
    )
    empty = [c for c in cells if not c.paper_ids]
    assert all(c.state == CellState.STRUCTURALLY_EMPTY for c in empty)
    assert top_gaps(cells) == []  # structurally empty excluded


def test_gap_score_ranks_pressured_demanded_cells() -> None:
    cells = build_gap_matrix(
        _papers(), "method", "dataset", now_year=2026,
        global_count_fn=lambda r, c: 5,
        demand_fn=lambda r, c: 1.0 if (r, c) == ("LSTM", "COMEX") else 0.0,
    )
    gaps = top_gaps(cells)
    assert gaps and gaps[0].row == "LSTM" and gaps[0].col == "COMEX"
    assert gaps[0].gap_score > 0


# --------------------------------------------------------------------------- claim clustering
def test_cluster_claims_groups_paraphrases() -> None:
    items = [
        ("LSTM with attention significantly improves forecasting accuracy over ARIMA",
         SupportingPaper("p1", frozenset({"doe"}), method="LSTM", year=2019)),
        ("GRU improves forecasting accuracy on industrial metals versus ARIMA",
         SupportingPaper("p2", frozenset({"zhang"}), method="GRU", year=2020)),
        ("GARCH captures volatility clustering better than rolling-window baselines",
         SupportingPaper("p3", frozenset({"hassan"}), method="GARCH", year=2017)),
    ]
    clusters = cluster_claims(items)
    # The two "improves forecasting accuracy over ARIMA" claims merge; GARCH stands alone.
    sizes = sorted(len(c.papers) for c in clusters)
    assert sizes == [1, 2]
    merged = max(clusters, key=lambda c: len(c.papers))
    assert {p.paper_id for p in merged.papers} == {"p1", "p2"}


def test_cluster_claims_does_not_merge_opposite_polarity() -> None:
    # "improves" and "shows no improvement" share subject tokens but must NOT cluster,
    # else a claim and its negation get conflated into one "finding".
    items = [
        ("Transformers improve forecasting accuracy over LSTM baselines",
         SupportingPaper("p1", frozenset({"a"}), year=2022)),
        ("Transformers show no significant improvement in forecasting accuracy over LSTM",
         SupportingPaper("p2", frozenset({"b"}), year=2023)),
    ]
    clusters = cluster_claims(items)
    assert len(clusters) == 2
    assert {c.polarity for c in clusters} == {1, -1}


def test_cluster_claims_feeds_known_knowns() -> None:
    # End to end: paraphrased convergent claims from independent authors -> a finding
    # that exact-text grouping would have missed entirely.
    items = [
        ("LSTM improves forecasting accuracy over ARIMA",
         SupportingPaper("p1", frozenset({"doe"}), method="LSTM", dataset="LME", year=2019)),
        ("GRU improves forecasting accuracy versus ARIMA",
         SupportingPaper("p2", frozenset({"zhang"}), method="GRU", dataset="COMEX", year=2020)),
    ]
    grouped = {c.canonical: c.papers for c in cluster_claims(items)}
    findings = consolidate_known_knowns(grouped)
    assert len(findings) == 1
    assert findings[0].independent_supports == 2
    assert findings[0].triangulated  # two methods + two datasets


# --------------------------------------------------------------------------- quadrants
def test_known_knowns_requires_independent_support() -> None:
    claims = {
        "LSTM beats ARIMA": [
            SupportingPaper("p1", frozenset({"doe"}), method="LSTM", dataset="LME", year=2023),
            SupportingPaper("p2", frozenset({"smith"}), method="VAR", dataset="COMEX", year=2024),
        ],
        "weak claim": [
            SupportingPaper("p3", frozenset({"doe"}), year=2020),
            SupportingPaper("p4", frozenset({"doe"}), year=2021),  # same author -> not independent
        ],
    }
    findings = consolidate_known_knowns(claims)
    texts = [f.claim for f in findings]
    assert "LSTM beats ARIMA" in texts
    assert "weak claim" not in texts  # only one independent group
    top = findings[0]
    assert top.independent_supports == 2 and top.triangulated


def test_known_knowns_excludes_contradicted() -> None:
    claims = {
        "c": [
            SupportingPaper("p1", frozenset({"a"}), year=2023),
            SupportingPaper("p2", frozenset({"b"}), year=2024),
        ]
    }
    assert consolidate_known_knowns(claims, contradicted={"c"}) == []


def test_cluster_open_problems() -> None:
    mentions = [
        OpenProblemMention("p1", "test on other commodities", [1.0, 0.0, 0.0], 2022),
        OpenProblemMention("p2", "evaluate on more commodities", [0.98, 0.02, 0.0], 2024),
        OpenProblemMention("p3", "improve interpretability", [0.0, 1.0, 0.0], 2023),
    ]
    clusters = cluster_open_problems(mentions, threshold=0.8, current_year=2026)
    assert len(clusters) == 2
    biggest = clusters[0]
    assert biggest.frequency == 2 and biggest.span_years == 2


def test_cross_community_transfer() -> None:
    sources = [("diffusion models", "A1", [1.0, 0.0, 0.0])]
    targets = [("B1", [0.99, 0.01, 0.0]), ("B2", [0.0, 1.0, 0.0])]
    # A1 and B1 are far apart in the graph; B2 is dissimilar.
    candidates = cross_community_transfer(
        sources, targets, distance_fn=lambda a, b: 5, sim_threshold=0.8, min_distance=3
    )
    assert len(candidates) == 1
    assert candidates[0].target_paper == "B1"
    assert candidates[0].method == "diffusion models"


def test_cross_community_transfer_skips_close_pairs() -> None:
    sources = [("m", "A1", [1.0, 0.0])]
    targets = [("B1", [1.0, 0.0])]
    # Graph distance is small -> not a 'next door' gap.
    candidates = cross_community_transfer(
        sources, targets, distance_fn=lambda a, b: 1, min_distance=3
    )
    assert candidates == []
