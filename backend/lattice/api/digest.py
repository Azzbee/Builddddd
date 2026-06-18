from __future__ import annotations

from fastapi import APIRouter, Depends

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/digest", tags=["digest"], dependencies=[Depends(require_auth)])


@router.post("/generate")
async def generate(c: Container = Depends(get_container)) -> dict[str, object]:
    """Build and store the current delta digest, then return it."""
    payload = await c.ingestion.generate_digest()
    await c.digests.add(payload)
    return payload


@router.get("/latest")
async def latest(c: Container = Depends(get_container)) -> dict[str, object]:
    return {"digest": await c.digests.latest()}


@router.get("/history")
async def history(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    return await c.digests.history()
