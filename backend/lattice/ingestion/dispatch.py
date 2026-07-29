"""Inline and arq-backed ingestion dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lattice.ingestion.models import IngestJob, JobStatus
from lattice.ingestion.service import IngestionService


class IngestionDispatcher(Protocol):
    asynchronous: bool

    async def submit(self, source_ref: str, pdf_bytes: bytes) -> IngestJob: ...
    async def retry(self, job_id: str) -> IngestJob | None: ...


class JobQueue(Protocol):
    async def enqueue_job(
        self,
        function: str,
        workspace_id: str,
        job_id: str,
        *,
        _job_id: str | None,
    ) -> object: ...


class JobRetryRejected(ValueError):
    """A job cannot be requeued in its current persisted state."""


class JobQueueUnavailable(RuntimeError):
    """Redis did not accept a staged ingestion job."""


def _validate_retry(job: IngestJob, max_attempts: int) -> None:
    if job.status not in (JobStatus.FAILED, JobStatus.PAUSED):
        raise JobRetryRejected(f"job status {job.status} cannot be retried")
    if job.attempts >= max_attempts:
        raise JobRetryRejected(f"job reached the attempt limit ({max_attempts})")
    if not job.retryable:
        raise JobRetryRejected(f"job failure {job.error_code or 'unknown'} is not retryable")


@dataclass
class InlineIngestionDispatcher:
    service: IngestionService
    asynchronous: bool = False

    async def submit(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        return await self.service.ingest_pdf(source_ref, pdf_bytes)

    async def retry(self, job_id: str) -> IngestJob | None:
        job = await self.service.jobs.get(job_id)
        if job is None:
            return None
        _validate_retry(job, self.service.settings.ingest_max_attempts)
        return await self.service.resume_job(job_id)


@dataclass
class ArqIngestionDispatcher:
    """Persist inputs first, then enqueue only small identifiers in Redis."""

    service: IngestionService
    redis: JobQueue
    asynchronous: bool = True

    async def submit(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        job = await self.service.stage_pdf(source_ref, pdf_bytes)
        if job.status != JobStatus.QUEUED:
            return job
        await self._enqueue_or_fail(job, deduplicate=True)
        return job

    async def retry(self, job_id: str) -> IngestJob | None:
        job = await self.service.jobs.get(job_id)
        if job is None:
            return None
        _validate_retry(job, self.service.settings.ingest_max_attempts)
        job.status = JobStatus.QUEUED
        job.error_code = None
        job.error_message = None
        job.retryable = False
        await self.service.jobs.save(job)
        # arq retains completed result keys for a while and rejects a new job
        # with the same explicit ID. Let arq create a fresh queue ID for every
        # manual retry, including cost-cap pauses that do not increment attempts.
        await self._enqueue_or_fail(job, deduplicate=False)
        return job

    async def _enqueue_or_fail(self, job: IngestJob, *, deduplicate: bool) -> None:
        try:
            await self._enqueue(job, deduplicate=deduplicate)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error_code = "queue_unavailable"
            job.error_message = "Redis did not accept the ingestion job"
            job.retryable = True
            await self.service.jobs.save(job)
            raise JobQueueUnavailable(job.error_message) from exc

    async def _enqueue(self, job: IngestJob, *, deduplicate: bool) -> None:
        queue_id = f"ingest:{job.job_id}:{job.attempts}" if deduplicate else None
        await self.redis.enqueue_job(
            "ingest_job_task",
            job.workspace_id,
            job.job_id,
            _job_id=queue_id,
        )
