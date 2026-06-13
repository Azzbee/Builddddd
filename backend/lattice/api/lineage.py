from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/lineage", tags=["lineage"], dependencies=[Depends(require_auth)])


@router.get("")
async def lineage(
    method: str = Query(..., description="Method family, e.g. 'lstm' or 'transformer'"),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Temporal DAG of how a method family evolved across the corpus."""
    return await c.ingestion.lineage(method)
