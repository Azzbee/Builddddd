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
from lattice.ingestion.models import JobStatus

log = get_logger("worker")

_RETRY_BACKOFF_BASE = 2
_MAX_RETRY_DELAY_S = 60


def _container(ctx: dict[str, Any], workspace_id: str) -> Container:
    containers: dict[str, Container] = ctx["containers"]
    if workspace_id not in containers:
        containers[workspace_id] = build_container(ctx["settings"], workspace_id=workspace_id)
    return containers[workspace_id]


async def ingest_job_task(ctx: dict[str, Any], workspace_id: str, job_id: str) -> dict[str, Any]:
    container = _container(ctx, workspace_id)
    job = await container.ingestion.resume_job(job_id)
    if job is None:
        raise RuntimeError(f"ingest job not found: {job_id}")
    if (
        job.status == JobStatus.FAILED
        and job.retryable
        and job.attempts < container.settings.ingest_max_attempts
    ):
        from arq import Retry

        delay = min(
            _MAX_RETRY_DELAY_S,
            _RETRY_BACKOFF_BASE ** max(0, job.attempts - 1),
        )
        log.warning("ingest.retry_scheduled", job_id=job_id, delay_s=delay)
        raise Retry(defer=delay)
    result: dict[str, Any] = job.model_dump(mode="json")
    return result


async def poll_arxiv(ctx: dict[str, Any]) -> int:
    """Cron: fetch recent arXiv papers, score against the corpus, queue matches."""
    settings = get_settings()
    container = _container(ctx, settings.workspace_id)
    await container.ingestion.hydrate()
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
    settings = get_settings()
    container = _container(ctx, settings.workspace_id)
    payload = await container.ingestion.generate_digest()
    await container.digests.add(payload)
    return payload


async def startup(ctx: dict[str, Any]) -> None:
    from lattice.api.deps import init_persistence

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    await init_persistence(settings)  # share Postgres/Neo4j with the API in prod
    ctx["settings"] = settings
    ctx["containers"] = {}
    log.info("worker.started", persistent=settings.persistent)


async def shutdown(ctx: dict[str, Any]) -> None:
    from lattice.api.deps import shutdown_persistence

    await shutdown_persistence()
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


def _redis_settings() -> Any:
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(get_settings().redis.url)


class WorkerSettings:
    """arq entrypoint: ``arq lattice.worker.WorkerSettings``."""

    functions = [ingest_job_task, poll_arxiv, generate_weekly_digest]
    on_startup = startup
    on_shutdown = shutdown
    cron_jobs = _cron_jobs()
    max_tries = get_settings().ingest_max_attempts
    redis_settings = _redis_settings()
