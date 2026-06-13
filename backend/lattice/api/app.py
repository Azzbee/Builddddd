"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lattice import __version__
from lattice.api import (
    contradictions,
    digest,
    graph,
    health,
    ingest,
    landscape,
    papers,
    query,
    reading,
    watch,
)
from lattice.config import get_settings
from lattice.core.logging import configure_logging


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

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(papers.router)
    app.include_router(graph.router)
    app.include_router(query.router)
    app.include_router(landscape.router)
    app.include_router(contradictions.router)
    app.include_router(reading.router)
    app.include_router(watch.router)
    app.include_router(digest.router)
    return app


app = create_app()
