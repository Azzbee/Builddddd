from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/watch", tags=["watch"], dependencies=[Depends(require_auth)])


class ApprovalAction(BaseModel):
    arxiv_id: str
    approve: bool


@router.get("/queue")
async def queue(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    """The approval queue of arXiv candidates similar to the corpus."""
    return await c.watch.pending()


@router.post("/approve")
async def approve(
    action: ApprovalAction, c: Container = Depends(get_container)
) -> dict[str, object]:
    status = "approved" if action.approve else "rejected"
    ok = await c.watch.set_status(action.arxiv_id, status)
    if not ok:
        return {"error": "not found"}
    return {"arxiv_id": action.arxiv_id, "status": status}
