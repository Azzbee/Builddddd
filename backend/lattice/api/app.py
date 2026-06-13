"""FastAPI application factory."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
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
from lattice.config import get_settings
from lattice.core.logging import configure_logging
from lattice.core.metrics import REGISTRY


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title="Lattice",
        version=__version__,
        description="A living knowledge graph for scientific literature.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production via config/reverse proxy
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
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

    @app.get("/metrics", include_in_schema=False)
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
