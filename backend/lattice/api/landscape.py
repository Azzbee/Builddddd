from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query

from lattice.api.deps import Container, get_container, require_auth
from lattice.core.logging import get_logger
from lattice.enrichment.openalex import OpenAlexClient
from lattice.landscape.coverage import DEFAULT_PROBE_LIMIT
from lattice.landscape.matrix import PaperFacets, build_gap_matrix, top_gaps
from lattice.landscape.momentum import momentum_score
from lattice.landscape.signals import DemandScorer, fetch_global_counts

router = APIRouter(prefix="/landscape", tags=["landscape"], dependencies=[Depends(require_auth)])
log = get_logger("api.landscape")

_FACETS = {"method", "dataset", "concept"}
#: Hard wall-clock budget for the (best-effort) OpenAlex global signal, so a slow
#: or unreachable API can never hang a landscape request. Degrades to no signal.
GLOBAL_SIGNAL_BUDGET_S = 6.0


async def _global_counts(c: Container, cells: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Fetch OpenAlex counts for ``cells`` under a strict time budget; {} on miss."""
    if not cells:
        return {}
    try:
        return await asyncio.wait_for(
            fetch_global_counts(OpenAlexClient(c.settings.enrichment), cells),
            timeout=GLOBAL_SIGNAL_BUDGET_S,
        )
    except (TimeoutError, Exception) as exc:  # best-effort: never block the request
        log.warning("landscape.global_signal_skipped", error=str(exc))
        return {}


async def _facets(c: Container) -> list[PaperFacets]:
    cards = await c.cards.all_cards()
    return [
        PaperFacets(
            paper_id=card.paper_id,
            year=card.year,
            methods=card.normalized_methods,
            datasets=card.normalized_datasets,
            concepts={d.lower() for d in card.domains},
        )
        for card in cards
    ]


@router.get("/matrix")
async def matrix(
    row: str = Query("method"),
    col: str = Query("dataset"),
    use_global: bool = Query(True, description="Query OpenAlex for the global signal"),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    if row not in _FACETS or col not in _FACETS:
        return {"error": f"facets must be in {sorted(_FACETS)}"}
    facets = await _facets(c)
    cards = await c.cards.all_cards()
    now_year = datetime.now(UTC).year

    # Local demand signal (always available) from corpus future-work/limitations.
    demand = DemandScorer.from_cards(c.chunk_embedder, cards)

    # Find empty cells to query the global signal for (Empty vs Blind-spot).
    rows = sorted({v for f in facets for v in f.facet(row)})
    cols = sorted({v for f in facets for v in f.facet(col)})
    populated = {(r, col_v) for f in facets for r in f.facet(row) for col_v in f.facet(col)}
    empties = [(r, cv) for r in rows for cv in cols if (r, cv) not in populated]

    global_counts = await _global_counts(c, empties) if use_global else {}

    cells = build_gap_matrix(
        facets,
        row,
        col,
        now_year=now_year,
        global_count_fn=lambda r, cv: global_counts.get((r, cv), 0),
        demand_fn=demand.score,
    )
    return {
        "row_facet": row,
        "col_facet": col,
        "global_signal": bool(global_counts),
        "cells": [cell.to_json() for cell in cells],
        "top_gaps": [cell.to_json() for cell in top_gaps(cells)],
    }


async def _empty_cell_global_counts(
    c: Container, facets: list[PaperFacets], row: str, col: str
) -> dict[tuple[str, str], int]:
    rows = sorted({v for f in facets for v in f.facet(row)})
    cols = sorted({v for f in facets for v in f.facet(col)})
    populated = {(r, cv) for f in facets for r in f.facet(row) for cv in f.facet(col)}
    empties = [(r, cv) for r in rows for cv in cols if (r, cv) not in populated]
    return await _global_counts(c, empties)


@router.get("/proposal")
async def proposal(
    row: str = Query(..., description="row facet value, e.g. a method"),
    col: str = Query(..., description="col facet value, e.g. a dataset"),
    row_facet: str = Query("method"),
    col_facet: str = Query("dataset"),
    use_global: bool = Query(
        False, description="Query OpenAlex for the Empty vs Blind-spot signal"
    ),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Generate a grounded research proposal for one (row, col) gap cell."""
    if row_facet not in _FACETS or col_facet not in _FACETS:
        return {"error": f"facets must be in {sorted(_FACETS)}"}
    global_count = 0
    if use_global:
        counts = await _global_counts(c, [(row.lower(), col.lower())])
        global_count = counts.get((row.lower(), col.lower()), 0)
    return await c.ingestion.research_proposal(
        row_facet, col_facet, row, col, global_count=global_count
    )


@router.get("/opportunities")
async def opportunities(
    row_facet: str = Query("method"),
    col_facet: str = Query("dataset"),
    limit: int = Query(5, ge=1, le=20),
    use_global: bool = Query(
        False, description="Query OpenAlex for the Empty vs Blind-spot signal"
    ),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Rank the corpus's top gaps and draft a full proposal for each."""
    if row_facet not in _FACETS or col_facet not in _FACETS:
        return {"error": f"facets must be in {sorted(_FACETS)}"}
    global_counts: dict[tuple[str, str], int] = {}
    if use_global:
        global_counts = await _empty_cell_global_counts(c, await _facets(c), row_facet, col_facet)
    return await c.ingestion.research_opportunities(
        row_facet, col_facet, limit=limit, global_counts=global_counts
    )


@router.get("/coverage")
async def coverage(
    row_facet: str = Query("method"),
    col_facet: str = Query("dataset"),
    limit: int = Query(DEFAULT_PROBE_LIMIT, ge=1, le=200, description="probe-bank size cap"),
    blind_spot_limit: int = Query(10, ge=1, le=50),
    use_global: bool = Query(
        False, description="Query OpenAlex for the Empty vs Blind-spot signal"
    ),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Question-coverage probing: the unknown-unknowns proxy.

    Asks the corpus a bank of questions it ought to answer (its own research
    questions, its open problems, and templated crossings of the highest-pressure
    gap cells), scores how well retrieval answers each, and ranks what it cannot.
    Retrieval-only and offline, so it needs no model key.
    """
    if row_facet not in _FACETS or col_facet not in _FACETS:
        return {"error": f"facets must be in {sorted(_FACETS)}"}
    global_counts: dict[tuple[str, str], int] = {}
    if use_global:
        global_counts = await _empty_cell_global_counts(c, await _facets(c), row_facet, col_facet)
    return await c.ingestion.question_coverage(
        row_facet,
        col_facet,
        limit=limit,
        blind_spot_limit=blind_spot_limit,
        global_counts=global_counts,
    )


@router.get("/momentum")
async def momentum(c: Container = Depends(get_container)) -> dict[str, object]:
    """Concept momentum from local corpus ingest years (global via OpenAlex in prod)."""
    cards = await c.cards.all_cards()
    counts: dict[str, dict[int, int]] = {}
    for card in cards:
        if card.year is None:
            continue
        for concept in card.domains:
            counts.setdefault(concept.lower(), {})
            counts[concept.lower()][card.year] = counts[concept.lower()].get(card.year, 0) + 1
    scored = [momentum_score(concept, ys) for concept, ys in counts.items()]
    scored.sort(key=lambda m: m.composite, reverse=True)
    return {"movers": [m.to_json() for m in scored]}


@router.get("/quadrants")
async def quadrants(c: Container = Depends(get_container)) -> dict[str, object]:
    """Epistemic quadrants: known knowns, known unknowns, unknown knowns."""
    return await c.ingestion.epistemic_quadrants()
