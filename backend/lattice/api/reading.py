from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/reading-queue", tags=["reading"], dependencies=[Depends(require_auth)])


@router.get("")
async def reading_queue(
    read: str = Query("", description="Comma-separated paper ids already read"),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Rank unread papers by expected information gain (neighborhood x novelty)."""
    read_ids = {x for x in read.split(",") if x.strip()}
    queue = await c.ingestion.reading_queue(read_ids)
    return {"read_count": len(read_ids), "queue": queue}
