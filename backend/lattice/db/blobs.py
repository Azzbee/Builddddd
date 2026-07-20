"""Raw PDF blob storage.

The reader UI deep-links into the original PDF (``/papers/{id}/pdf#page=N``), so
the source bytes are persisted at ingest time keyed by ``paper_id``. The in-memory
store keeps the system fully runnable offline; the Postgres ``bytea`` store is the
production backend and shares the same :class:`BlobStore` protocol, so it drops in
unchanged. Page counts are computed once at store time (pypdf, best-effort) so the
UI and citation deep-links know the document length without re-parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lattice.core.hashing import content_hash as _hash
from lattice.core.logging import get_logger

log = get_logger("db.blobs")


@dataclass
class BlobMeta:
    """Lightweight metadata for a stored PDF (no bytes)."""

    paper_id: str
    size: int
    pages: int
    content_hash: str

    def to_json(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "available": True,
            "size": self.size,
            "pages": self.pages,
            "content_hash": self.content_hash,
        }


def pdf_page_count(data: bytes) -> int:
    """Best-effort page count. Returns 0 when pypdf is unavailable or the bytes
    are not a parseable PDF (e.g. synthetic demo bytes), never raising."""
    try:
        import io

        from pypdf import PdfReader

        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:  # pragma: no cover - defensive; exercised via 0-page path
        return 0


class BlobStore(Protocol):
    async def put(
        self, paper_id: str, data: bytes, *, content_hash: str | None = None
    ) -> BlobMeta: ...
    async def get(self, paper_id: str) -> bytes | None: ...
    async def meta(self, paper_id: str) -> BlobMeta | None: ...


class InMemoryBlobStore:
    """Dependency-free blob store backed by a dict. Used offline and in tests."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._meta: dict[str, BlobMeta] = {}

    async def put(self, paper_id: str, data: bytes, *, content_hash: str | None = None) -> BlobMeta:
        meta = BlobMeta(
            paper_id=paper_id,
            size=len(data),
            pages=pdf_page_count(data),
            content_hash=content_hash or _hash(data),
        )
        self._data[paper_id] = data
        self._meta[paper_id] = meta
        return meta

    async def get(self, paper_id: str) -> bytes | None:
        return self._data.get(paper_id)

    async def meta(self, paper_id: str) -> BlobMeta | None:
        return self._meta.get(paper_id)


# SQL kept as named constants so it is reusable, greppable, and statically
# validated (see tests/test_sql.py, which parses each with the Postgres dialect).
SQL_UPSERT_BLOB = """
INSERT INTO pdf_blobs (paper_id, workspace_id, data, size, pages, content_hash, created_at)
VALUES ($1, $2, $3, $4, $5, $6, now())
ON CONFLICT (paper_id) DO UPDATE SET
    data = EXCLUDED.data, size = EXCLUDED.size, pages = EXCLUDED.pages,
    content_hash = EXCLUDED.content_hash
"""
SQL_GET_BLOB = "SELECT data FROM pdf_blobs WHERE workspace_id = $1 AND paper_id = $2"
SQL_BLOB_META = (
    "SELECT paper_id, size, pages, content_hash FROM pdf_blobs "
    "WHERE workspace_id = $1 AND paper_id = $2"
)

#: All PgBlobStore statements, for static validation.
BLOB_SQL = {
    "upsert_blob": SQL_UPSERT_BLOB,
    "get_blob": SQL_GET_BLOB,
    "blob_meta": SQL_BLOB_META,
}


@dataclass
class PgBlobStore:  # pragma: no cover - requires Postgres
    """Postgres ``bytea``-backed blob store. Same interface as the in-memory one."""

    pool: Any
    workspace_id: str = "default"

    async def put(self, paper_id: str, data: bytes, *, content_hash: str | None = None) -> BlobMeta:
        meta = BlobMeta(
            paper_id=paper_id,
            size=len(data),
            pages=pdf_page_count(data),
            content_hash=content_hash or _hash(data),
        )
        async with self.pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_BLOB,
                paper_id,
                self.workspace_id,
                data,
                meta.size,
                meta.pages,
                meta.content_hash,
            )
        return meta

    async def get(self, paper_id: str) -> bytes | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchval(SQL_GET_BLOB, self.workspace_id, paper_id)
        return bytes(row) if row is not None else None

    async def meta(self, paper_id: str) -> BlobMeta | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(SQL_BLOB_META, self.workspace_id, paper_id)
        if row is None:
            return None
        return BlobMeta(
            paper_id=row["paper_id"],
            size=row["size"],
            pages=row["pages"],
            content_hash=row["content_hash"],
        )
