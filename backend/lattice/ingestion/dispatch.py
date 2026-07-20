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
        _job_id: str,
    ) -> object: ...


@dataclass
class InlineIngestionDispatcher:
    service: IngestionService
    asynchronous: bool = False

    async def submit(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        return await self.service.ingest_pdf(source_ref, pdf_bytes)

    async def retry(self, job_id: str) -> IngestJob | None:
        return await self.service.resume_job(job_id)


@dataclass
class ArqIngestionDispatcher:
    """Persist inputs first, then enqueue only small identifiers in Redis."""

    service: IngestionService
    redis: JobQueue
    asynchronous: bool = True

    async def submit(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        job = await self.service.stage_pdf(source_ref, pdf_bytes)
        if job.status in (JobStatus.SUCCEEDED, JobStatus.DUPLICATE):
            return job
        await self._enqueue(job)
        return job

    async def retry(self, job_id: str) -> IngestJob | None:
        job = await self.service.jobs.get(job_id)
        if job is None:
            return None
        if job.status in (JobStatus.SUCCEEDED, JobStatus.DUPLICATE):
            return job
        if job.attempts >= self.service.settings.ingest_max_attempts:
            return job
        job.status = JobStatus.QUEUED
        job.error_code = None
        job.error_message = None
        await self.service.jobs.save(job)
        await self._enqueue(job)
        return job

    async def _enqueue(self, job: IngestJob) -> None:
        queue_id = f"ingest:{job.job_id}:{job.attempts}"
        await self.redis.enqueue_job(
            "ingest_job_task",
            job.workspace_id,
            job.job_id,
            _job_id=queue_id,
        )
