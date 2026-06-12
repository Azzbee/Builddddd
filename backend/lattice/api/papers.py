from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/papers", tags=["papers"], dependencies=[Depends(require_auth)])


class ReviewUpdate(BaseModel):
    needs_review: bool | None = None
    problem_statement: str | None = None


@router.get("")
async def list_papers(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    cards = await c.cards.all_cards()
    return [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "year": p.year,
            "authors": [a.name for a in p.authors],
            "paper_type": str(p.paper_type),
            "confidence": p.confidence,
            "needs_review": p.needs_review,
        }
        for p in cards
    ]


@router.get("/{paper_id}")
async def get_paper(paper_id: str, c: Container = Depends(get_container)) -> dict[str, object]:
    card = await c.cards.get_card(paper_id)
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")
    return card


@router.get("/{paper_id}/neighbors")
async def neighbors(paper_id: str, c: Container = Depends(get_container)) -> dict[str, object]:
    snapshot = await c.ingestion.graph_snapshot()
    edges = [e for e in snapshot.edges if paper_id in (e.source, e.target)]
    edges.sort(key=lambda e: e.weight, reverse=True)
    return {"paper_id": paper_id, "neighbors": [e.to_json() for e in edges]}


@router.post("/{paper_id}/review")
async def update_review(
    paper_id: str, update: ReviewUpdate, c: Container = Depends(get_container)
) -> dict[str, object]:
    """Human-in-the-loop correction. Writes back to the card."""
    card = await c.cards.get(paper_id)
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")
    if update.needs_review is not None:
        card.needs_review = update.needs_review
    if update.problem_statement is not None:
        card.problem_statement = update.problem_statement
    await c.cards.put_card(card)
    return {"paper_id": paper_id, "needs_review": card.needs_review}
