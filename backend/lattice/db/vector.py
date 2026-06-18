"""Vector store: chunk embeddings for ANN candidate generation and hybrid search.

The in-memory implementation is a real, correct store (cosine ANN + a BM25-style
keyword score fused with vector score) so the system runs end-to-end with no
Postgres. The pgvector implementation shares the same interface for production.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from lattice.core.hashing import normalize_text
from lattice.rag.models import ChunkHit


@dataclass
class VectorRecord:
    chunk_id: str
    paper_id: str
    workspace_id: str
    title: str
    section_title: str
    text: str
    embedding: list[float]
    evidence_location: str | None = None


class VectorStore(Protocol):
    async def upsert_chunks(self, records: list[VectorRecord]) -> None: ...
    async def ann_search_papers(
        self, embedding: list[float], k: int, *, workspace_id: str, exclude_paper: str | None = None
    ) -> list[tuple[str, float]]: ...
    async def hybrid_search(
        self, query_text: str, query_embedding: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkHit]: ...


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class InMemoryVectorStore:
    """A correct, dependency-free vector store with hybrid retrieval."""

    def __init__(self, hybrid_vector_weight: float = 0.6) -> None:
        self._records: dict[str, VectorRecord] = {}
        self._vec_w = hybrid_vector_weight
        # Document frequencies for the BM25-ish keyword score.
        self._df: Counter[str] = Counter()

    async def upsert_chunks(self, records: list[VectorRecord]) -> None:
        for r in records:
            if r.chunk_id in self._records:
                self._decr_df(self._records[r.chunk_id].text)
            self._records[r.chunk_id] = r
            self._incr_df(r.text)

    def _incr_df(self, text: str) -> None:
        for tok in set(normalize_text(text).split()):
            self._df[tok] += 1

    def _decr_df(self, text: str) -> None:
        for tok in set(normalize_text(text).split()):
            if self._df[tok] > 0:
                self._df[tok] -= 1

    async def ann_search_papers(
        self, embedding: list[float], k: int, *, workspace_id: str, exclude_paper: str | None = None
    ) -> list[tuple[str, float]]:
        """Return up to k nearest *papers* (best chunk per paper) by cosine."""
        q = np.asarray(embedding, dtype=float)
        best: dict[str, float] = {}
        for r in self._records.values():
            if r.workspace_id != workspace_id or r.paper_id == exclude_paper:
                continue
            sim = _cos(q, np.asarray(r.embedding, dtype=float))
            if sim > best.get(r.paper_id, -1.0):
                best[r.paper_id] = sim
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def _keyword_score(self, query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        n_docs = max(1, len({r.chunk_id for r in self._records.values()}))
        doc_tokens = normalize_text(text).split()
        if not doc_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        score = 0.0
        for tok in query_tokens:
            if tok not in tf:
                continue
            idf = math.log(1 + n_docs / (1 + self._df.get(tok, 0)))
            score += idf * (tf[tok] / len(doc_tokens))
        return score

    async def hybrid_search(
        self, query_text: str, query_embedding: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        ws = (filters or {}).get("workspace_id")
        q = np.asarray(query_embedding, dtype=float)
        qtokens = normalize_text(query_text).split()
        scored: list[tuple[float, VectorRecord]] = []
        for r in self._records.values():
            if ws is not None and r.workspace_id != ws:
                continue
            vec_score = _cos(q, np.asarray(r.embedding, dtype=float))
            kw_score = self._keyword_score(qtokens, r.text)
            fused = self._vec_w * vec_score + (1 - self._vec_w) * _squash(kw_score)
            scored.append((fused, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ChunkHit(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                title=r.title,
                section_title=r.section_title,
                text=r.text,
                score=score,
                evidence_location=r.evidence_location,
            )
            for score, r in scored[:k]
        ]


def _squash(x: float) -> float:
    return math.tanh(x)


# SQL kept as named constants so it is reusable, greppable, and statically
# validated (see tests/test_sql.py, which parses each with the Postgres dialect).
SQL_UPSERT_CHUNK = """
INSERT INTO chunks (chunk_id, paper_id, workspace_id, title, section_title,
                    text, embedding, evidence_location)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text
"""

SQL_ANN_PAPERS = """
SELECT paper_id, MAX(1 - (embedding <=> $1)) AS sim
FROM chunks
WHERE workspace_id = $2 AND ($3::text IS NULL OR paper_id <> $3)
GROUP BY paper_id ORDER BY sim DESC LIMIT $4
"""

SQL_HYBRID_SEARCH = """
SELECT chunk_id, paper_id, title, section_title, text, evidence_location,
       (1 - (embedding <=> $1)) AS vec_score,
       ts_rank(to_tsvector('english', text), plainto_tsquery('english', $2)) AS kw_score
FROM chunks WHERE workspace_id = $3
ORDER BY ($4 * (1 - (embedding <=> $1))) +
         ((1 - $4) * ts_rank(to_tsvector('english', text), plainto_tsquery('english', $2))) DESC
LIMIT $5
"""

#: All PgVectorStore statements, for static validation.
PGVECTOR_SQL = {
    "upsert_chunk": SQL_UPSERT_CHUNK,
    "ann_papers": SQL_ANN_PAPERS,
    "hybrid_search": SQL_HYBRID_SEARCH,
}


async def create_pg_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> Any:  # pragma: no cover
    """Create an asyncpg pool with pgvector type codecs registered per connection."""
    import asyncpg
    from pgvector.asyncpg import register_vector

    async def _init(conn: Any) -> None:
        await register_vector(conn)

    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init)


@dataclass
class PgVectorStore:  # pragma: no cover - requires Postgres + pgvector
    """pgvector-backed store. Same interface as InMemoryVectorStore; SQL above."""

    pool: Any
    workspace_id: str = "default"
    hybrid_vector_weight: float = 0.6

    async def upsert_chunks(self, records: list[VectorRecord]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(
                SQL_UPSERT_CHUNK,
                [
                    (r.chunk_id, r.paper_id, r.workspace_id, r.title, r.section_title,
                     r.text, r.embedding, r.evidence_location)
                    for r in records
                ],
            )

    async def ann_search_papers(
        self, embedding: list[float], k: int, *, workspace_id: str, exclude_paper: str | None = None
    ) -> list[tuple[str, float]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(SQL_ANN_PAPERS, embedding, workspace_id, exclude_paper, k)
            return [(r["paper_id"], float(r["sim"])) for r in rows]

    async def hybrid_search(
        self, query_text: str, query_embedding: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        ws = (filters or {}).get("workspace_id", self.workspace_id)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                SQL_HYBRID_SEARCH, query_embedding, query_text, ws, self.hybrid_vector_weight, k
            )
            return [
                ChunkHit(
                    chunk_id=r["chunk_id"], paper_id=r["paper_id"], title=r["title"],
                    section_title=r["section_title"], text=r["text"],
                    score=float(r["vec_score"]), evidence_location=r["evidence_location"],
                )
                for r in rows
            ]
