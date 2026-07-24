from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from lattice import __version__
from lattice.api.deps import list_workspaces, persistence_health, require_auth

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
async def ready(response: Response) -> dict[str, str | dict[str, bool]]:
    checks = await persistence_health()
    is_ready = all(checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if is_ready else "not_ready", "checks": checks}
