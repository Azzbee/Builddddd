from __future__ import annotations

from fastapi import APIRouter, Depends

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/digest", tags=["digest"], dependencies=[Depends(require_auth)])


@router.post("/generate")
async def generate(c: Container = Depends(get_container)) -> dict[str, object]:
    """Build and store the current delta digest, then return it."""
    payload = await c.ingestion.generate_digest()
    c._digests.append(payload)
    return payload


@router.get("/latest")
async def latest(c: Container = Depends(get_container)) -> dict[str, object]:
    if not c._digests:
        return {"digest": None}
    return {"digest": c._digests[-1]}


@router.get("/history")
async def history(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    return list(reversed(c._digests))
