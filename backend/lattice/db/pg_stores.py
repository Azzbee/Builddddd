"""Postgres-backed card and job stores (production persistence).

Implement the same CorpusStore / JobStore protocols as the in-memory versions so
they drop into the ingestion service unchanged. The full PaperCard is stored as
JSONB in ``papers.card`` with denormalized identity columns for dedup and listing.
Covered by tests/integration/test_pg_stores_integration.py against a real Postgres.
"""

from __future__ import annotations

import json
from typing import Any

from lattice.db.cards import StoredFeatures
from lattice.extraction.schemas import PaperCard
from lattice.ingestion.dedup import CorpusIndex, PaperIdentity
from lattice.ingestion.models import IngestJob

SQL_UPSERT_PAPER = """
INSERT INTO papers (paper_id, workspace_id, title, authors, year, venue, doi, arxiv_id,
                    s2_paper_id, card, confidence, needs_review, specter, aspects,
                    reference_ids, content_hash, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, now())
ON CONFLICT (paper_id) DO UPDATE SET
    title = EXCLUDED.title, authors = EXCLUDED.authors, year = EXCLUDED.year,
    venue = EXCLUDED.venue, doi = EXCLUDED.doi, arxiv_id = EXCLUDED.arxiv_id,
    s2_paper_id = EXCLUDED.s2_paper_id, card = EXCLUDED.card,
    confidence = EXCLUDED.confidence, needs_review = EXCLUDED.needs_review,
    specter = COALESCE(EXCLUDED.specter, papers.specter),
    aspects = COALESCE(EXCLUDED.aspects, papers.aspects),
    reference_ids = COALESCE(EXCLUDED.reference_ids, papers.reference_ids),
    content_hash = COALESCE(EXCLUDED.content_hash, papers.content_hash),
    updated_at = now()
"""

SQL_GET_CARD = "SELECT card FROM papers WHERE workspace_id = $1 AND paper_id = $2"
SQL_ALL_CARDS = "SELECT card FROM papers WHERE workspace_id = $1 ORDER BY updated_at DESC"
SQL_CORPUS_INDEX = (
    "SELECT paper_id, title, authors, doi, arxiv_id, content_hash "
    "FROM papers WHERE workspace_id = $1"
)
# card holds methods/datasets; specter is the paper-level SPECTER vector; aspects
# are the problem/methodology/results vectors. Together they rehydrate incremental
# linking at full similarity fidelity after a restart. Superseded papers are
# excluded so they never re-enter the candidate pool.
SQL_LOAD_FEATURES = (
    "SELECT paper_id, card, specter, aspects, reference_ids FROM papers "
    "WHERE workspace_id = $1 AND superseded_by IS NULL"
)
SQL_MARK_SUPERSEDED = (
    "UPDATE papers SET superseded_by = $3, updated_at = now() "
    "WHERE workspace_id = $1 AND paper_id = $2"
)
SQL_SUPERSEDED_MAP = (
    "SELECT paper_id, superseded_by FROM papers "
    "WHERE workspace_id = $1 AND superseded_by IS NOT NULL"
)

SQL_UPSERT_JOB = """
INSERT INTO ingest_jobs (job_id, workspace_id, source_type, source_ref, content_hash,
                         paper_id, stage, status, error_code, error_message, retryable,
                         attempts, cost_usd, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, now())
ON CONFLICT (job_id) DO UPDATE SET
    content_hash = EXCLUDED.content_hash, paper_id = EXCLUDED.paper_id,
    stage = EXCLUDED.stage, status = EXCLUDED.status, error_code = EXCLUDED.error_code,
    error_message = EXCLUDED.error_message, retryable = EXCLUDED.retryable,
    attempts = EXCLUDED.attempts,
    cost_usd = EXCLUDED.cost_usd, updated_at = now()
"""

SQL_GET_JOB = "SELECT * FROM ingest_jobs WHERE workspace_id = $1 AND job_id = $2"
SQL_ALL_JOBS = "SELECT * FROM ingest_jobs WHERE workspace_id = $1 ORDER BY created_at DESC"

SQL_ENQUEUE_WATCH = """
INSERT INTO watch_queue (workspace_id, arxiv_id, title, similarity, nearest_paper_id, status)
VALUES ($1, $2, $3, $4, $5, 'pending')
ON CONFLICT (workspace_id, arxiv_id) DO NOTHING
"""
SQL_PENDING_WATCH = (
    "SELECT arxiv_id, title, similarity, nearest_paper_id, status FROM watch_queue "
    "WHERE workspace_id = $1 AND status = 'pending' ORDER BY similarity DESC"
)
SQL_SET_WATCH_STATUS = (
    "UPDATE watch_queue SET status = $3 WHERE workspace_id = $1 AND arxiv_id = $2"
)
SQL_ADD_DIGEST = "INSERT INTO digests (workspace_id, period_label, report) VALUES ($1, $2, $3)"
SQL_LATEST_DIGEST = (
    "SELECT report FROM digests WHERE workspace_id = $1 ORDER BY created_at DESC LIMIT 1"
)
SQL_DIGEST_HISTORY = "SELECT report FROM digests WHERE workspace_id = $1 ORDER BY created_at DESC"

#: All statements, for static SQL validation in tests/test_sql.py.
PG_STORE_SQL = {
    "upsert_paper": SQL_UPSERT_PAPER,
    "get_card": SQL_GET_CARD,
    "all_cards": SQL_ALL_CARDS,
    "corpus_index": SQL_CORPUS_INDEX,
    "load_features": SQL_LOAD_FEATURES,
    "mark_superseded": SQL_MARK_SUPERSEDED,
    "superseded_map": SQL_SUPERSEDED_MAP,
    "upsert_job": SQL_UPSERT_JOB,
    "get_job": SQL_GET_JOB,
    "all_jobs": SQL_ALL_JOBS,
    "enqueue_watch": SQL_ENQUEUE_WATCH,
    "pending_watch": SQL_PENDING_WATCH,
    "set_watch_status": SQL_SET_WATCH_STATUS,
    "add_digest": SQL_ADD_DIGEST,
    "latest_digest": SQL_LATEST_DIGEST,
    "digest_history": SQL_DIGEST_HISTORY,
}


def _loads(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _vector_to_list(value: Any) -> list[float] | None:
    """Normalize a pgvector column value to a plain list of floats.

    The codec's return type varies by pgvector-python version: ndarray/list in
    older releases, a non-iterable ``Vector`` (with ``to_list()``) from 0.5.0.
    Handle all of them so an unpinned dependency bump can't break rehydration.
    """
    if value is None:
        return None
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [float(x) for x in to_list()]
    return [float(x) for x in value]


class PgCardStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def put_card(
        self,
        card: PaperCard,
        specter: list[float] | None = None,
        aspects: dict[str, list[float]] | None = None,
        references: list[str] | None = None,
        content_hash: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_PAPER,
                card.paper_id,
                self._ws,
                card.title,
                json.dumps([a.model_dump(mode="json") for a in card.authors]),
                card.year,
                card.venue,
                card.doi,
                card.arxiv_id,
                card.s2_paper_id,
                card.model_dump_json(),
                card.confidence,
                card.needs_review,
                specter,
                json.dumps(aspects) if aspects is not None else None,
                json.dumps(references) if references is not None else None,
                content_hash,
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

    async def load_features(self) -> list[StoredFeatures]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_LOAD_FEATURES, self._ws)
        out: list[StoredFeatures] = []
        for r in rows:
            card = PaperCard.model_validate(_loads(r["card"]))
            specter = _vector_to_list(r["specter"])
            aspects = _loads(r["aspects"]) if r["aspects"] is not None else None
            refs = _loads(r["reference_ids"]) if r["reference_ids"] is not None else None
            out.append(
                StoredFeatures(
                    paper_id=r["paper_id"],
                    specter=specter,
                    methods=card.normalized_methods,
                    datasets=card.normalized_datasets,
                    aspects=aspects if isinstance(aspects, dict) else None,
                    references=frozenset(refs) if isinstance(refs, list) else frozenset(),
                    external_ids=frozenset(card.external_ids),
                )
            )
        return out

    async def mark_superseded(self, paper_id: str, superseded_by: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(SQL_MARK_SUPERSEDED, self._ws, paper_id, superseded_by)

    async def superseded_map(self) -> dict[str, str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_SUPERSEDED_MAP, self._ws)
        return {r["paper_id"]: r["superseded_by"] for r in rows}


class PgJobStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def save(self, job: IngestJob) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_JOB,
                job.job_id,
                self._ws,
                str(job.source_type),
                job.source_ref,
                job.content_hash,
                job.paper_id,
                str(job.stage),
                str(job.status),
                job.error_code,
                job.error_message,
                job.retryable,
                job.attempts,
                job.cost_usd,
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


class PgWatchStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def enqueue(self, items: list[dict[str, Any]]) -> int:
        added = 0
        async with self._pool.acquire() as conn:
            for item in items:
                result = await conn.execute(
                    SQL_ENQUEUE_WATCH,
                    self._ws,
                    str(item.get("arxiv_id")),
                    item.get("title", ""),
                    float(item.get("similarity", 0.0)),
                    item.get("nearest_paper_id"),
                )
                if result.endswith("1"):  # "INSERT 0 1" when a row was added
                    added += 1
        return added

    async def pending(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_PENDING_WATCH, self._ws)
        return [dict(r) for r in rows]

    async def set_status(self, arxiv_id: str, status: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(SQL_SET_WATCH_STATUS, self._ws, arxiv_id, status)
        return bool(result.endswith("1"))


class PgDigestStore:  # pragma: no cover - exercised by integration tests
    def __init__(self, pool: Any, workspace_id: str = "default") -> None:
        self._pool = pool
        self._ws = workspace_id

    async def add(self, report: dict[str, Any]) -> None:
        label = str((report.get("report") or {}).get("period_label", "")) or "digest"
        async with self._pool.acquire() as conn:
            await conn.execute(SQL_ADD_DIGEST, self._ws, label, json.dumps(report))

    async def latest(self) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(SQL_LATEST_DIGEST, self._ws)
        return _loads(row) if row else None

    async def history(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL_DIGEST_HISTORY, self._ws)
        return [_loads(r["report"]) for r in rows]


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
        retryable=row["retryable"],
        attempts=row["attempts"],
        cost_usd=row["cost_usd"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
