"""arq worker: async, resumable background jobs.

Defines the ingestion task plus scheduled cron jobs for the arXiv watcher and the
weekly digest. The worker shares the same wired container as the API, so jobs and
HTTP requests operate on identical services. Requires Redis (the ``api`` extra).
"""

from __future__ import annotations

from typing import Any

from lattice.api.deps import build_container
from lattice.config import get_settings
from lattice.core.logging import configure_logging, get_logger
from lattice.enrichment.arxiv_watcher import ArxivWatcher

log = get_logger("worker")


async def ingest_pdf_task(ctx: dict[str, Any], source_ref: str, pdf_bytes: bytes) -> dict[str, Any]:
    container = ctx["container"]
    job = await container.ingestion.ingest_pdf(source_ref, pdf_bytes)
    await container.jobs.save(job)
    result: dict[str, Any] = job.model_dump(mode="json")
    return result


async def poll_arxiv(ctx: dict[str, Any]) -> int:
    """Cron: fetch recent arXiv papers, score against the corpus, queue matches."""
    container = ctx["container"]
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
    existing = {item["arxiv_id"] for item in container._watch_queue}
    added = 0
    for s in scored:
        if s.candidate.arxiv_id in existing:
            continue
        container._watch_queue.append({**s.to_json(), "status": "pending"})
        added += 1
    log.info("watcher.queued", added=added)
    return added


async def generate_weekly_digest(ctx: dict[str, Any]) -> dict[str, Any]:
    """Cron: build and persist the weekly delta digest."""
    from datetime import UTC, datetime

    from lattice.digest.weekly import DigestInput, build_digest, render_markdown

    container = ctx["container"]
    cards = await container.cards.all_cards()
    snapshot = await container.ingestion.graph_snapshot()
    label = datetime.now(UTC).strftime("%Y-W%V")
    report = build_digest(
        DigestInput(period_label=label, new_papers=[], new_edges=len(snapshot.edges))
    )
    payload = {"report": report.to_json(), "markdown": render_markdown(report), "papers": len(cards)}
    container._digests.append(payload)
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
