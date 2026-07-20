from __future__ import annotations

from fastapi import APIRouter, Depends

from lattice import __version__
from lattice.api.deps import list_workspaces, require_auth

router = APIRouter(tags=["health"])


@router.get("/workspaces", dependencies=[Depends(require_auth)])
async def workspaces() -> dict[str, list[str]]:
    """List corpora (workspaces) that have been touched this process."""
    return {"workspaces": list_workspaces()}


@router.get("/health")
@router.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "lattice", "version": __version__}


@router.get("/readyz")
async def ready() -> dict[str, str]:
    # Liveness only here; deep readiness checks (Neo4j/PG/Redis pings) belong in
    # the worker/ops layer. The API serving requests is itself the signal.
    return {"status": "ready"}
