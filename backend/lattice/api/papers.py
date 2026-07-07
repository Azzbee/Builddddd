from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/papers", tags=["papers"], dependencies=[Depends(require_auth)])


class ReviewUpdate(BaseModel):
    needs_review: bool | None = None
    problem_statement: str | None = None


class SummarizeRequest(BaseModel):
    paper_ids: list[str]


@router.get("")
async def list_papers(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    cards = await c.cards.all_cards()
    superseded = await c.cards.superseded_map()
    return [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "year": p.year,
            "authors": [a.name for a in p.authors],
            "paper_type": str(p.paper_type),
            "confidence": p.confidence,
            "needs_review": p.needs_review,
            # Superseded versions stay listed (supersede, never delete) so the UI
            # can badge them and link to the successor.
            "superseded_by": superseded.get(p.paper_id),
        }
        for p in cards
    ]


@router.post("/summarize")
async def summarize(
    req: SummarizeRequest, c: Container = Depends(get_container)
) -> dict[str, object]:
    """Brief for a hand-selected subset of papers (explorer lasso-select)."""
    return await c.ingestion.summarize_papers(req.paper_ids)


@router.get("/{paper_id}")
async def get_paper(paper_id: str, c: Container = Depends(get_container)) -> dict[str, object]:
    card = await c.cards.get_card(paper_id)
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")
    superseded = await c.cards.superseded_map()
    card["superseded_by"] = superseded.get(paper_id)
    return card


@router.get("/{paper_id}/pdf/meta")
async def pdf_meta(paper_id: str, c: Container = Depends(get_container)) -> dict[str, object]:
    """Whether a source PDF is stored, and its page count, for the reader UI."""
    meta = await c.blobs.meta(paper_id)
    if meta is None:
        return {"paper_id": paper_id, "available": False, "pages": 0, "size": 0}
    return meta.to_json()


@router.get("/{paper_id}/pdf")
async def pdf(paper_id: str, c: Container = Depends(get_container)) -> Response:
    """Stream the stored source PDF (inline) so the web reader can embed it."""
    data = await c.blobs.get(paper_id)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no stored PDF for this paper")
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{paper_id}.pdf"'},
    )


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
