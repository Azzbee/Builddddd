"""Durable stage artifacts for resumable ingestion.

Jobs record which stage finished. This store records the data needed by the next
stage, so a worker restart can continue without reparsing the source PDF or
repeating paid model calls.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from lattice.extraction.schemas import PaperCard
from lattice.ingestion.chunker import Chunk
from lattice.ingestion.models import ParsedDocument
from lattice.ingestion.pipeline import PipelineContext

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def _json_extra(extra: dict[str, object]) -> dict[str, JsonValue]:
    """Keep persistable stage data while dropping process-local collaborators."""
    out: dict[str, JsonValue] = {}
    for key, value in extra.items():
        try:
            out[key] = TypeAdapter(JsonValue).validate_python(value)
        except ValidationError:
            continue
    return out


class IngestArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_pdf: bytes | None = None
    document: ParsedDocument | None = None
    card: PaperCard | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    extra: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def capture(cls, ctx: PipelineContext) -> IngestArtifacts:
        return cls(
            raw_pdf=ctx.raw_pdf,
            document=ctx.document,
            card=ctx.card,
            chunks=ctx.chunks,
            extra=_json_extra(ctx.extra),
        )

    def restore(self, ctx: PipelineContext) -> None:
        ctx.raw_pdf = self.raw_pdf
        ctx.document = self.document
        ctx.card = self.card
        ctx.chunks = list(self.chunks)
        ctx.extra.update(self.extra)


class IngestArtifactStore(Protocol):
    async def save(self, job_id: str, artifacts: IngestArtifacts) -> None: ...
    async def get(self, job_id: str) -> IngestArtifacts | None: ...
    async def save_context(self, ctx: PipelineContext) -> None: ...


class InMemoryIngestArtifactStore:
    def __init__(self) -> None:
        self._artifacts: dict[str, IngestArtifacts] = {}

    async def save(self, job_id: str, artifacts: IngestArtifacts) -> None:
        self._artifacts[job_id] = artifacts.model_copy(deep=True)

    async def get(self, job_id: str) -> IngestArtifacts | None:
        value = self._artifacts.get(job_id)
        return value.model_copy(deep=True) if value is not None else None

    async def save_context(self, ctx: PipelineContext) -> None:
        await self.save(ctx.job.job_id, IngestArtifacts.capture(ctx))


SQL_UPSERT_INGEST_ARTIFACTS = """
INSERT INTO ingest_artifacts
    (job_id, workspace_id, raw_pdf, document, card, chunks, extra, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, now())
ON CONFLICT (job_id) DO UPDATE SET
    raw_pdf = COALESCE(EXCLUDED.raw_pdf, ingest_artifacts.raw_pdf),
    document = COALESCE(EXCLUDED.document, ingest_artifacts.document),
    card = COALESCE(EXCLUDED.card, ingest_artifacts.card),
    chunks = EXCLUDED.chunks,
    extra = EXCLUDED.extra,
    updated_at = now()
"""

SQL_GET_INGEST_ARTIFACTS = """
SELECT raw_pdf, document, card, chunks, extra
FROM ingest_artifacts
WHERE workspace_id = $1 AND job_id = $2
"""

INGEST_ARTIFACT_SQL = {
    "upsert_ingest_artifacts": SQL_UPSERT_INGEST_ARTIFACTS,
    "get_ingest_artifacts": SQL_GET_INGEST_ARTIFACTS,
}


def _decode_json(value: Any, default: object) -> object:
    if value is None:
        return default
    return json.loads(value) if isinstance(value, str) else value


class PgIngestArtifactStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def save(self, job_id: str, artifacts: IngestArtifacts) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_INGEST_ARTIFACTS,
                job_id,
                self._ws,
                artifacts.raw_pdf,
                artifacts.document.model_dump_json() if artifacts.document else None,
                artifacts.card.model_dump_json() if artifacts.card else None,
                json.dumps([chunk.model_dump(mode="json") for chunk in artifacts.chunks]),
                json.dumps(_JSON_OBJECT.validate_python(artifacts.extra)),
            )

    async def get(self, job_id: str) -> IngestArtifacts | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(SQL_GET_INGEST_ARTIFACTS, self._ws, job_id)
        if row is None:
            return None
        return IngestArtifacts.model_validate(
            {
                "raw_pdf": row["raw_pdf"],
                "document": _decode_json(row["document"], None),
                "card": _decode_json(row["card"], None),
                "chunks": _decode_json(row["chunks"], []),
                "extra": _decode_json(row["extra"], {}),
            }
        )

    async def save_context(self, ctx: PipelineContext) -> None:
        await self.save(ctx.job.job_id, IngestArtifacts.capture(ctx))
