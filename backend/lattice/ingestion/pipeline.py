"""The ingestion orchestrator: a resumable state machine.

The pipeline walks a job through the happy-path stages, running an injected
handler for each. Handlers are async and mutate a shared :class:`PipelineContext`.
Failures map to typed statuses:

* :class:`DuplicateError` -> status DUPLICATE (idempotent no-op)
* :class:`CostCapExceeded` -> status PAUSED (resume later from the same stage)
* any other :class:`LatticeError` -> status FAILED with a machine-readable code

Because ``job.stage`` is always the last *completed* stage and is persisted after
each step, a crash resumes exactly where it left off. Linking is the final stage
and touches the graph last, so a crash before it leaves no paper in the graph; its
writes are idempotent MERGEs, so a re-run of a partially-linked paper converges
rather than duplicating. (Linking is not a single cross-store transaction, so this
relies on idempotency, not rollback.)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from lattice.core.errors import CostCapExceeded, DuplicateError, LatticeError
from lattice.core.logging import get_logger
from lattice.extraction.schemas import PaperCard
from lattice.ingestion.chunker import Chunk
from lattice.ingestion.models import IngestJob, JobStage, JobStatus, ParsedDocument, next_stage

log = get_logger("ingestion.pipeline")


@dataclass
class PipelineContext:
    """Mutable carrier for artifacts produced as the job advances."""

    job: IngestJob
    raw_pdf: bytes | None = None
    document: ParsedDocument | None = None
    card: PaperCard | None = None
    chunks: list[Chunk] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)


StageFn = Callable[[PipelineContext], Awaitable[None]]


class JobStore(Protocol):
    """Persistence boundary for job state (implemented over Postgres in prod)."""

    async def save(self, job: IngestJob) -> None: ...


class ArtifactStore(Protocol):
    async def save_context(self, ctx: PipelineContext) -> None: ...


class _NullStore:
    async def save(self, job: IngestJob) -> None:  # pragma: no cover - trivial
        return None


class IngestionPipeline:
    """Drives a job to completion using per-stage handlers."""

    def __init__(
        self,
        handlers: dict[JobStage, StageFn],
        store: JobStore | None = None,
        artifact_store: ArtifactStore | None = None,
        max_attempts: int = 3,
    ):
        self._handlers = handlers
        self._store = store or _NullStore()
        self._artifact_store = artifact_store
        self._max_attempts = max_attempts

    async def _persist(self, job: IngestJob) -> None:
        job.updated_at = datetime.now(UTC)
        await self._store.save(job)

    async def run(self, ctx: PipelineContext) -> IngestJob:
        """Advance the job until it reaches a terminal or paused state."""
        job = ctx.job
        if job.status in (JobStatus.PAUSED, JobStatus.FAILED):
            if job.attempts >= self._max_attempts:
                return job
            job.status = JobStatus.QUEUED
        await self._persist(job)
        while not job.is_terminal:
            target = next_stage(job.stage)
            if target is None or target == JobStage.DONE:
                job.stage = JobStage.DONE
                job.status = JobStatus.SUCCEEDED
                await self._persist(job)
                log.info("ingest.done", job_id=job.job_id, paper_id=job.paper_id)
                break

            handler = self._handlers.get(target)
            if handler is None:
                raise LatticeError(f"no handler registered for stage {target}")

            job.status = JobStatus.RUNNING
            await self._persist(job)
            try:
                await handler(ctx)
            except DuplicateError as exc:
                job.status = JobStatus.DUPLICATE
                job.error_code = exc.code
                job.error_message = str(exc)
                if not job.paper_id:
                    job.paper_id = ctx.extra.get("duplicate_of")  # type: ignore[assignment]
                await self._persist(job)
                log.info("ingest.duplicate", job_id=job.job_id, of=job.paper_id)
                break
            except CostCapExceeded as exc:
                job.status = JobStatus.PAUSED
                job.error_code = exc.code
                job.error_message = str(exc)
                await self._persist(job)
                log.warning("ingest.paused", job_id=job.job_id, stage=target, reason=exc.code)
                break
            except LatticeError as exc:
                job.attempts += 1
                job.error_code = exc.code
                job.error_message = str(exc)
                job.status = JobStatus.FAILED
                await self._persist(job)
                log.error(
                    "ingest.failed",
                    job_id=job.job_id,
                    stage=target,
                    code=exc.code,
                    retryable=exc.retryable,
                    attempts=job.attempts,
                )
                break
            else:
                cost = ctx.extra.get("cost")
                usage = getattr(cost, "job_usage", None)
                if usage is not None:
                    job.cost_usd = float(getattr(usage, "cost_usd", job.cost_usd))
                job.stage = target
                job.error_code = None
                job.error_message = None
                if self._artifact_store is not None:
                    await self._artifact_store.save_context(ctx)
                await self._persist(job)
                log.info("ingest.stage_complete", job_id=job.job_id, stage=target)

        return job
