"""Research Gap Matrix (generalized): cross any two PaperCard facets.

Each cell is state-classified (saturated/active/contested/stale/empty/blind-spot)
and empty cells are gap-scored:

    gap_score = feasibility x adjacency_pressure x demand_signal

The crucial Empty vs Blind-spot distinction uses a global signal (OpenAlex count):
an empty cell is only a real research gap if the world has not filled it either.
Feasibility, demand, and global-count are injected callables so the builder is
pure and testable offline; in production they wrap an LLM judge and OpenAlex.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

# Facet accessor: returns the set of values a paper has for a facet.
FacetFn = Callable[["PaperFacets"], set[str]]

SATURATION_MIN = 4
STALE_YEARS = 4


class CellState(StrEnum):
    SATURATED = "saturated"
    ACTIVE = "active"
    CONTESTED = "contested"
    STALE = "stale"
    EMPTY = "empty"
    BLIND_SPOT = "blind_spot"
    STRUCTURALLY_EMPTY = "structurally_empty"


@dataclass
class PaperFacets:
    paper_id: str
    year: int | None = None
    methods: set[str] = field(default_factory=set)
    datasets: set[str] = field(default_factory=set)
    concepts: set[str] = field(default_factory=set)

    def facet(self, name: str) -> set[str]:
        return {"method": self.methods, "dataset": self.datasets, "concept": self.concepts}[name]


@dataclass
class MatrixCell:
    row: str
    col: str
    paper_ids: list[str]
    latest_year: int | None
    sparkline: dict[int, int]
    state: CellState
    global_count: int = 0
    gap_score: float = 0.0
    feasibility: float = 1.0
    adjacency_pressure: float = 0.0
    demand_signal: float = 0.0

    @property
    def paper_count(self) -> int:
        return len(self.paper_ids)

    def to_json(self) -> dict[str, object]:
        return {
            "row": self.row,
            "col": self.col,
            "paper_count": self.paper_count,
            "paper_ids": self.paper_ids,
            "latest_year": self.latest_year,
            "sparkline": self.sparkline,
            "state": str(self.state),
            "global_count": self.global_count,
            "gap_score": round(self.gap_score, 4),
            "components": {
                "feasibility": round(self.feasibility, 3),
                "adjacency_pressure": round(self.adjacency_pressure, 3),
                "demand_signal": round(self.demand_signal, 3),
            },
        }


def _values(papers: list[PaperFacets], facet: str) -> list[str]:
    vals: set[str] = set()
    for p in papers:
        vals |= p.facet(facet)
    return sorted(vals)


def build_gap_matrix(
    papers: list[PaperFacets],
    row_facet: str,
    col_facet: str,
    *,
    now_year: int,
    contradiction_pairs: set[frozenset[str]] | None = None,
    global_count_fn: Callable[[str, str], int] | None = None,
    feasibility_fn: Callable[[str, str], float] | None = None,
    demand_fn: Callable[[str, str], float] | None = None,
    blind_spot_min_global: int = 25,
) -> list[MatrixCell]:
    """Build the full set of matrix cells crossing two facets."""
    contradiction_pairs = contradiction_pairs or set()
    global_count_fn = global_count_fn or (lambda r, c: 0)
    feasibility_fn = feasibility_fn or (lambda r, c: 1.0)
    demand_fn = demand_fn or (lambda r, c: 0.0)

    rows = _values(papers, row_facet)
    cols = _values(papers, col_facet)

    # Row/column marginal activity for adjacency pressure.
    row_activity = {r: sum(1 for p in papers if r in p.facet(row_facet)) for r in rows}
    col_activity = {c: sum(1 for p in papers if c in p.facet(col_facet)) for c in cols}
    max_marginal = max([*row_activity.values(), *col_activity.values(), 1])

    cells: list[MatrixCell] = []
    for r in rows:
        for c in cols:
            in_cell = [
                p
                for p in papers
                if r in p.facet(row_facet) and c in p.facet(col_facet)
            ]
            sparkline: dict[int, int] = {}
            for p in in_cell:
                if p.year is not None:
                    sparkline[p.year] = sparkline.get(p.year, 0) + 1
            latest = max((p.year for p in in_cell if p.year is not None), default=None)

            if in_cell:
                state = _classify_populated(in_cell, latest, now_year, r, c, contradiction_pairs)
                cells.append(
                    MatrixCell(
                        row=r,
                        col=c,
                        paper_ids=[p.paper_id for p in in_cell],
                        latest_year=latest,
                        sparkline=sparkline,
                        state=state,
                    )
                )
                continue

            # Empty cell: feasibility gate, then global signal, then gap score.
            feasibility = feasibility_fn(r, c)
            if feasibility <= 0.05:
                cells.append(
                    MatrixCell(r, c, [], None, {}, CellState.STRUCTURALLY_EMPTY, feasibility=feasibility)
                )
                continue
            global_count = global_count_fn(r, c)
            adjacency = (row_activity[r] + col_activity[c]) / (2 * max_marginal)
            demand = demand_fn(r, c)
            gap_score = feasibility * adjacency * demand
            state = CellState.BLIND_SPOT if global_count >= blind_spot_min_global else CellState.EMPTY
            cells.append(
                MatrixCell(
                    row=r,
                    col=c,
                    paper_ids=[],
                    latest_year=None,
                    sparkline={},
                    state=state,
                    global_count=global_count,
                    gap_score=gap_score,
                    feasibility=feasibility,
                    adjacency_pressure=adjacency,
                    demand_signal=demand,
                )
            )
    return cells


def _classify_populated(
    in_cell: list[PaperFacets],
    latest: int | None,
    now_year: int,
    row: str,
    col: str,
    contradiction_pairs: set[frozenset[str]],
) -> CellState:
    ids = {p.paper_id for p in in_cell}
    for pair in contradiction_pairs:
        if pair <= ids:
            return CellState.CONTESTED
    if latest is not None and latest < now_year - STALE_YEARS:
        return CellState.STALE
    if latest is not None and latest >= now_year - 1 and len(in_cell) >= 2:
        return CellState.ACTIVE
    if len(in_cell) >= SATURATION_MIN:
        return CellState.SATURATED
    return CellState.ACTIVE


def top_gaps(cells: list[MatrixCell], limit: int = 10) -> list[MatrixCell]:
    """Empty/blind-spot cells ranked by gap_score (excludes structurally empty)."""
    candidates = [
        c for c in cells if c.state in (CellState.EMPTY, CellState.BLIND_SPOT)
    ]
    candidates.sort(key=lambda c: c.gap_score, reverse=True)
    return candidates[:limit]
