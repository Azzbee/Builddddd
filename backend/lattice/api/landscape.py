from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query

from lattice.api.deps import Container, get_container, require_auth
from lattice.landscape.matrix import PaperFacets, build_gap_matrix, top_gaps
from lattice.landscape.momentum import momentum_score

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
    c: Container = Depends(get_container),
) -> dict[str, object]:
    if row not in _FACETS or col not in _FACETS:
        return {"error": f"facets must be in {sorted(_FACETS)}"}
    facets = await _facets(c)
    now_year = datetime.now(UTC).year
    cells = build_gap_matrix(facets, row, col, now_year=now_year)
    return {
        "row_facet": row,
        "col_facet": col,
        "cells": [cell.to_json() for cell in cells],
        "top_gaps": [cell.to_json() for cell in top_gaps(cells)],
    }


@router.get("/momentum")
async def momentum(c: Container = Depends(get_container)) -> dict[str, object]:
    """Concept momentum from local corpus ingest years (global signals via OpenAlex
    in production). Computes a scorecard per domain concept."""
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
