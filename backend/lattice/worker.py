"""arq worker: async, resumable background jobs.

Defines the ingestion task plus scheduled cron jobs for the arXiv watcher and the
weekly digest. The worker shares the same wired container as the API, so jobs and
HTTP requests operate on identical services. Requires Redis (the ``api`` extra).
"""

from __future__ import annotations

from typing import Any

from lattice.api.deps import Container, build_container
from lattice.config import get_settings
from lattice.core.logging import configure_logging, get_logger
from lattice.enrichment.arxiv_watcher import ArxivWatcher

log = get_logger("worker")


async def ingest_pdf_task(ctx: dict[str, Any], source_ref: str, pdf_bytes: bytes) -> dict[str, Any]:
    container: Container = ctx["container"]
    job = await container.ingestion.ingest_pdf(source_ref, pdf_bytes)
    await container.jobs.save(job)
    result: dict[str, Any] = job.model_dump(mode="json")
    return result


async def poll_arxiv(ctx: dict[str, Any]) -> int:
    """Cron: fetch recent arXiv papers, score against the corpus, queue matches."""
    container: Container = ctx["container"]
    settings = get_settings()
    watcher = ArxivWatcher(settings.enrichment)
    try:
        results = await watcher.fetch_recent(settings.watcher.arxiv_categories, max_results=50)
    except Exception as exc:
        log.warning("watcher.fetch_failed", error=str(exc))
        return 0

    from lattice.watch import score_candidates

    corpus_vectors = [
        (pid, feat.specter.tolist())
        for pid, feat in container.ingestion._features.items()
        if feat.specter is not None
    ]
    # Embed candidates in the SAME space as the corpus paper vectors (SPECTER).
    scored = score_candidates(
        results,
        corpus_vectors,
        embedder=container.ingestion.specter.text_backend,
        floor=settings.watcher.similarity_floor,
    )
    added = await container.watch.enqueue([s.to_json() for s in scored])
    log.info("watcher.queued", added=added)
    return added


async def generate_weekly_digest(ctx: dict[str, Any]) -> dict[str, Any]:
    """Cron: build and persist the weekly delta digest."""
    container: Container = ctx["container"]
    payload = await container.ingestion.generate_digest()
    await container.digests.add(payload)
    return payload


async def startup(ctx: dict[str, Any]) -> None:
    from lattice.api.deps import init_persistence

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    await init_persistence(settings)  # share Postgres/Neo4j with the API in prod
    ctx["container"] = build_container(settings)
    log.info("worker.started", persistent=settings.persistent)


async def shutdown(ctx: dict[str, Any]) -> None:
    log.info("worker.stopped")


def _cron_jobs() -> list[Any]:
    try:
        from arq import cron
    except ImportError:  # pragma: no cover
        return []
    return [
        cron(poll_arxiv, hour=set(range(0, 24, 6)), minute=0),  # every 6h
        cron(generate_weekly_digest, weekday=0, hour=8, minute=0),  # Monday 08:00
    ]


class WorkerSettings:
    """arq entrypoint: ``arq lattice.worker.WorkerSettings``."""

    functions = [ingest_pdf_task, poll_arxiv, generate_weekly_digest]
    on_startup = startup
    on_shutdown = shutdown
    cron_jobs = _cron_jobs()

    @staticmethod
    def redis_settings() -> Any:  # pragma: no cover - needs redis
        from arq.connections import RedisSettings

        return RedisSettings.from_dsn(get_settings().redis.url)
