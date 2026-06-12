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
    return [item for item in c._watch_queue if item.get("status") == "pending"]


@router.post("/approve")
async def approve(action: ApprovalAction, c: Container = Depends(get_container)) -> dict[str, object]:
    for item in c._watch_queue:
        if item.get("arxiv_id") == action.arxiv_id:
            item["status"] = "approved" if action.approve else "rejected"
            return {"arxiv_id": action.arxiv_id, "status": item["status"]}
    return {"error": "not found"}
