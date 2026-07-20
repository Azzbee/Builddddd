from __future__ import annotations

from typing import Any

import pytest
from lattice.api.deps import build_container
from lattice.config import Settings
from lattice.demo import demo_pdf_bytes
from lattice.ingestion.dispatch import (
    ArqIngestionDispatcher,
    JobQueueUnavailable,
    JobRetryRejected,
)
from lattice.ingestion.models import JobStatus


class FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail = fail

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> object:
        self.calls.append((function, args, kwargs))
        if self.fail:
            raise ConnectionError("redis unavailable")
        return object()


async def test_arq_dispatch_stages_pdf_and_enqueues_only_identifiers() -> None:
    container = build_container(Settings(demo_mode=True, persistent=False))
    redis = FakeRedis()
    dispatcher = ArqIngestionDispatcher(container.ingestion, redis)

    job = await dispatcher.submit("paper.pdf", demo_pdf_bytes("lstm_copper.pdf"))

    assert job.status == JobStatus.QUEUED
    saved = await container.artifacts.get(job.job_id)
    assert saved is not None and saved.raw_pdf is not None
    assert redis.calls == [
        (
            "ingest_job_task",
            ("default", job.job_id),
            {"_job_id": f"ingest:{job.job_id}:0"},
        )
    ]
    assert demo_pdf_bytes("lstm_copper.pdf") not in redis.calls[0][1]


async def test_arq_retry_is_bounded_and_uses_new_attempt_id() -> None:
    settings = Settings(demo_mode=True, ingest_max_attempts=2)
    container = build_container(settings)
    redis = FakeRedis()
    dispatcher = ArqIngestionDispatcher(container.ingestion, redis)
    job = await container.ingestion.stage_pdf("paper.pdf", demo_pdf_bytes("lstm_copper.pdf"))
    job.status = JobStatus.FAILED
    job.retryable = True
    job.attempts = 1
    await container.jobs.save(job)

    retried = await dispatcher.retry(job.job_id)
    assert retried is not None and retried.status == JobStatus.QUEUED
    assert redis.calls[0][2]["_job_id"] == f"ingest:{job.job_id}:1"

    retried.status = JobStatus.FAILED
    retried.retryable = True
    retried.attempts = 2
    await container.jobs.save(retried)
    with pytest.raises(JobRetryRejected, match="attempt limit"):
        await dispatcher.retry(job.job_id)
    assert len(redis.calls) == 1


async def test_arq_retry_rejects_terminal_failure() -> None:
    container = build_container(Settings(demo_mode=True))
    dispatcher = ArqIngestionDispatcher(container.ingestion, FakeRedis())
    job = await container.ingestion.stage_pdf("bad.pdf", b"not a pdf")
    job.status = JobStatus.FAILED
    job.error_code = "corrupted_pdf"
    job.retryable = False
    await container.jobs.save(job)

    with pytest.raises(JobRetryRejected, match="corrupted_pdf is not retryable"):
        await dispatcher.retry(job.job_id)


async def test_arq_submit_does_not_requeue_existing_terminal_job() -> None:
    container = build_container(Settings(demo_mode=True))
    redis = FakeRedis()
    dispatcher = ArqIngestionDispatcher(container.ingestion, redis)
    pdf = demo_pdf_bytes("lstm_copper.pdf")
    job = await container.ingestion.stage_pdf("paper.pdf", pdf)
    job.status = JobStatus.FAILED
    job.error_code = "corrupted_pdf"
    await container.jobs.save(job)

    existing = await dispatcher.submit("paper.pdf", pdf)

    assert existing.status == JobStatus.FAILED
    assert redis.calls == []


async def test_arq_queue_failure_is_persisted_and_retryable() -> None:
    container = build_container(Settings(demo_mode=True))
    redis = FakeRedis(fail=True)
    dispatcher = ArqIngestionDispatcher(container.ingestion, redis)

    with pytest.raises(JobQueueUnavailable, match="Redis did not accept"):
        await dispatcher.submit("paper.pdf", demo_pdf_bytes("lstm_copper.pdf"))

    jobs = await container.jobs.all_jobs()
    assert len(jobs) == 1
    failed = jobs[0]
    assert failed.status == JobStatus.FAILED
    assert failed.error_code == "queue_unavailable"
    assert failed.retryable is True

    redis.fail = False
    retried = await dispatcher.retry(failed.job_id)
    assert retried is not None and retried.status == JobStatus.QUEUED
