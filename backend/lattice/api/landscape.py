from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query

from lattice.api.deps import Container, get_container, require_auth
from lattice.enrichment.openalex import OpenAlexClient
from lattice.landscape.matrix import PaperFacets, build_gap_matrix, top_gaps
from lattice.landscape.momentum import momentum_score
from lattice.landscape.signals import DemandScorer, fetch_global_counts

router = APIRouter(prefix="/landscape", tags=["landscape"], dependencies=[Depends(require_auth)])

_FACETS = {"method", "dataset", "concept"}


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
    populated = {
        (r, col_v)
        for f in facets
        for r in f.facet(row)
        for col_v in f.facet(col)
    }
    empties = [(r, cv) for r in rows for cv in cols if (r, cv) not in populated]

    global_counts: dict[tuple[str, str], int] = {}
    if use_global and empties:
        try:
            openalex = OpenAlexClient(c.settings.enrichment)
            global_counts = await fetch_global_counts(openalex, empties)
        except Exception:
            global_counts = {}

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
