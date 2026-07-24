"""FastAPI application factory."""

from __future__ import annotations

import hmac
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from lattice import __version__
from lattice.api import (
    contradictions,
    digest,
    export,
    graph,
    health,
    ingest,
    landscape,
    lineage,
    papers,
    query,
    reading,
    watch,
)
from lattice.api.deps import require_auth
from lattice.config import get_settings
from lattice.core.logging import configure_logging
from lattice.core.metrics import REGISTRY
from lattice.core.ratelimit import SlidingWindowLimiter

CallNext = Callable[[Request], Awaitable[Response]]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        from lattice.api.deps import get_container, init_persistence, shutdown_persistence

        await init_persistence(settings)
        if settings.demo_mode:
            from lattice.demo import load_demo

            await load_demo(get_container(settings.workspace_id))
        yield
        await shutdown_persistence()

    app = FastAPI(
        title="Lattice",
        version=__version__,
        description="A living knowledge graph for scientific literature.",
        lifespan=lifespan,
        docs_url=None if settings.environment == "prod" else "/docs",
        redoc_url=None if settings.environment == "prod" else "/redoc",
        openapi_url=None if settings.environment == "prod" else "/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    limiter = SlidingWindowLimiter(settings.rate_limit_per_min, window_s=60.0)
    _EXEMPT = {"/health", "/healthz", "/readyz", "/metrics"}

    @app.middleware("http")
    async def _rate_limit_middleware(request: Request, call_next: CallNext) -> Response:
        if request.url.path in _EXEMPT:
            return await call_next(request)
        auth = request.headers.get("authorization")
        expected = f"Bearer {settings.auth_token}" if settings.auth_token else None
        authenticated = (
            expected is not None and auth is not None and hmac.compare_digest(auth, expected)
        )
        key = (
            "authenticated"
            if authenticated
            else (request.client.host if request.client else "anonymous")
        )
        allowed, retry_after = limiter.allow(key)
        if not allowed:
            REGISTRY.inc("lattice_http_rate_limited_total", help="Rate-limited requests")
            return Response(
                content='{"detail":"rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return await call_next(request)

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next: CallNext) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        # Use the route template (not the raw path) to bound label cardinality.
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        labels = {"method": request.method, "path": path, "status": str(response.status_code)}
        REGISTRY.inc("lattice_http_requests_total", labels, help="HTTP requests")
        REGISTRY.observe(
            "lattice_http_request_duration_seconds",
            elapsed,
            {"method": request.method, "path": path},
            help="HTTP request latency",
        )
        return response

    @app.get(
        "/metrics",
        include_in_schema=False,
        dependencies=[Depends(require_auth)],
    )
    async def metrics() -> Response:
        return Response(content=REGISTRY.render(), media_type="text/plain; version=0.0.4")

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(papers.router)
    app.include_router(graph.router)
    app.include_router(query.router)
    app.include_router(landscape.router)
    app.include_router(contradictions.router)
    app.include_router(reading.router)
    app.include_router(lineage.router)
    app.include_router(export.router)
    app.include_router(watch.router)
    app.include_router(digest.router)
    return app


app = create_app()
