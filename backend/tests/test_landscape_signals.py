from __future__ import annotations

from lattice.embeddings.base import HashingEmbedder
from lattice.extraction.schemas import Methodology, PaperCard
from lattice.landscape.matrix import CellState, PaperFacets, build_gap_matrix
from lattice.landscape.signals import DemandScorer, cell_description, fetch_global_counts


def _card(pid: str, future_work: list[str]) -> PaperCard:
    return PaperCard(
        paper_id=pid,
        title=pid,
        methodology=Methodology(approach_summary="x"),
        future_work=future_work,
    )


def test_cell_description() -> None:
    assert cell_description("lstm", "comex") == "lstm applied to comex"


def test_demand_scorer_higher_for_demanded_cell() -> None:
    embedder = HashingEmbedder(dim=512)
    cards = [
        _card("p1", ["future work should test diffusion models on supply disruption forecasting"]),
        _card("p2", ["evaluate interpretability of attention layers"]),
    ]
    scorer = DemandScorer.from_cards(embedder, cards)
    demanded = scorer.score("diffusion models", "supply disruption forecasting")
    unrelated = scorer.score("quantum", "chromodynamics")
    assert 0.0 <= demanded <= 1.0
    assert demanded > unrelated


def test_demand_scorer_empty_corpus() -> None:
    scorer = DemandScorer.from_cards(HashingEmbedder(dim=16), [])
    assert scorer.score("a", "b") == 0.0


class FakeOpenAlex:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.calls = 0

    async def works_count(self, search: str) -> int:
        self.calls += 1
        return self.counts.get(search, 0)


async def test_fetch_global_counts() -> None:
    oa = FakeOpenAlex({"lstm comex": 120})
    counts = await fetch_global_counts(oa, [("lstm", "comex"), ("var", "lme")])
    assert counts[("lstm", "comex")] == 120
    assert counts.get(("var", "lme"), 0) == 0
    assert oa.calls == 2


async def test_fetch_global_counts_respects_cap() -> None:
    oa = FakeOpenAlex({})
    cells = [(f"m{i}", "d") for i in range(10)]
    await fetch_global_counts(oa, cells, cap=3)
    assert oa.calls == 3


async def test_fetch_global_counts_graceful_without_client() -> None:
    assert await fetch_global_counts(object(), [("a", "b")]) == {}


def test_matrix_blind_spot_via_global_count() -> None:
    papers = [PaperFacets("p1", 2024, methods={"lstm"}, datasets={"lme"})]
    # (lstm, comex) is empty locally; high global count -> blind spot, not empty.
    cells = build_gap_matrix(
        papers, "method", "dataset", now_year=2026,
        global_count_fn=lambda r, c: 200 if (r, c) == ("lstm", "comex") else 0,
    )
    by = {(c.row, c.col): c for c in cells}
    # Only "lme" is a known dataset; build a 2-dataset matrix by adding a paper.
    papers2 = [
        PaperFacets("p1", 2024, methods={"lstm"}, datasets={"lme"}),
        PaperFacets("p2", 2024, methods={"var"}, datasets={"comex"}),
    ]
    cells2 = build_gap_matrix(
        papers2, "method", "dataset", now_year=2026,
        global_count_fn=lambda r, c: 200 if (r, c) == ("lstm", "comex") else 0,
    )
    by2 = {(c.row, c.col): c for c in cells2}
    assert by2[("lstm", "comex")].state == CellState.BLIND_SPOT
    assert by2[("var", "lme")].state in (CellState.EMPTY, CellState.BLIND_SPOT)
    assert by  # sanity
