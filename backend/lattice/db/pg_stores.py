"""Postgres-backed card and job stores (production persistence).

Implement the same CorpusStore / JobStore protocols as the in-memory versions so
they drop into the ingestion service unchanged. The full PaperCard is stored as
JSONB in ``papers.card`` with denormalized identity columns for dedup and listing.
Covered by tests/integration/test_pg_stores_integration.py against a real Postgres.
"""

from __future__ import annotations

import json
from typing import Any

from lattice.extraction.schemas import PaperCard
from lattice.ingestion.dedup import CorpusIndex, PaperIdentity
from lattice.ingestion.models import IngestJob

SQL_UPSERT_PAPER = """
INSERT INTO papers (paper_id, workspace_id, title, authors, year, venue, doi, arxiv_id,
                    s2_paper_id, card, confidence, needs_review, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now())
ON CONFLICT (paper_id) DO UPDATE SET
    title = EXCLUDED.title, authors = EXCLUDED.authors, year = EXCLUDED.year,
    venue = EXCLUDED.venue, doi = EXCLUDED.doi, arxiv_id = EXCLUDED.arxiv_id,
    s2_paper_id = EXCLUDED.s2_paper_id, card = EXCLUDED.card,
    confidence = EXCLUDED.confidence, needs_review = EXCLUDED.needs_review,
    updated_at = now()
"""

SQL_GET_CARD = "SELECT card FROM papers WHERE workspace_id = $1 AND paper_id = $2"
SQL_ALL_CARDS = "SELECT card FROM papers WHERE workspace_id = $1 ORDER BY updated_at DESC"
SQL_CORPUS_INDEX = (
    "SELECT paper_id, title, authors, doi, arxiv_id, content_hash "
    "FROM papers WHERE workspace_id = $1"
)

SQL_UPSERT_JOB = """
INSERT INTO ingest_jobs (job_id, workspace_id, source_type, source_ref, content_hash,
                         paper_id, stage, status, error_code, error_message, attempts,
                         cost_usd, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, now())
ON CONFLICT (job_id) DO UPDATE SET
    content_hash = EXCLUDED.content_hash, paper_id = EXCLUDED.paper_id,
    stage = EXCLUDED.stage, status = EXCLUDED.status, error_code = EXCLUDED.error_code,
    error_message = EXCLUDED.error_message, attempts = EXCLUDED.attempts,
    cost_usd = EXCLUDED.cost_usd, updated_at = now()
"""

SQL_GET_JOB = "SELECT * FROM ingest_jobs WHERE workspace_id = $1 AND job_id = $2"
SQL_ALL_JOBS = "SELECT * FROM ingest_jobs WHERE workspace_id = $1 ORDER BY created_at DESC"

#: All statements, for static SQL validation in tests/test_sql.py.
PG_STORE_SQL = {
    "upsert_paper": SQL_UPSERT_PAPER,
    "get_card": SQL_GET_CARD,
    "all_cards": SQL_ALL_CARDS,
    "corpus_index": SQL_CORPUS_INDEX,
    "upsert_job": SQL_UPSERT_JOB,
    "get_job": SQL_GET_JOB,
    "all_jobs": SQL_ALL_JOBS,
}


def _loads(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class PgCardStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def put_card(self, card: PaperCard) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_PAPER,
                card.paper_id, self._ws, card.title,
                json.dumps([a.model_dump(mode="json") for a in card.authors]),
                card.year, card.venue, card.doi, card.arxiv_id, card.s2_paper_id,
                card.model_dump_json(), card.confidence, card.needs_review,
            )

    async def get_card(self, paper_id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(SQL_GET_CARD, self._ws, paper_id)
        return _loads(row) if row else None

    async def get(self, paper_id: str) -> PaperCard | None:
        data = await self.get_card(paper_id)
        return PaperCard.model_validate(data) if data else None

    async def all_cards(self) -> list[PaperCard]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_ALL_CARDS, self._ws)
        return [PaperCard.model_validate(_loads(r["card"])) for r in rows]

    async def corpus_index(self) -> CorpusIndex:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_CORPUS_INDEX, self._ws)
        idents = [
            PaperIdentity(
                paper_id=r["paper_id"],
                title=r["title"],
                authors=[a.get("name", "") for a in _loads(r["authors"]) or []],
                doi=r["doi"],
                arxiv_id=r["arxiv_id"],
                content_hash=r["content_hash"],
            )
            for r in rows
        ]
        return CorpusIndex.from_identities(idents)


class PgJobStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def save(self, job: IngestJob) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_JOB,
                job.job_id, self._ws, str(job.source_type), job.source_ref,
                job.content_hash, job.paper_id, str(job.stage), str(job.status),
                job.error_code, job.error_message, job.attempts, job.cost_usd,
                job.created_at,
            )

    async def get(self, job_id: str) -> IngestJob | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(SQL_GET_JOB, self._ws, job_id)
        return _row_to_job(row) if row else None

    async def all_jobs(self) -> list[IngestJob]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_ALL_JOBS, self._ws)
        return [_row_to_job(r) for r in rows]


def _row_to_job(row: Any) -> IngestJob:
    return IngestJob(
        job_id=row["job_id"],
        workspace_id=row["workspace_id"],
        source_type=row["source_type"],
        source_ref=row["source_ref"],
        content_hash=row["content_hash"],
        paper_id=row["paper_id"],
        stage=row["stage"],
        status=row["status"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        attempts=row["attempts"],
        cost_usd=row["cost_usd"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
