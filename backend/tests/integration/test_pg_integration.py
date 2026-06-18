"""Live Postgres + pgvector integration tests.

Gated on ``LATTICE_TEST_PG_DSN`` (set in CI by a service container). Exercises the
real PgVectorStore against a real pgvector instance: schema apply, chunk upsert,
ANN paper search, and hybrid (vector + full-text) search.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DSN = os.environ.get("LATTICE_TEST_PG_DSN")
if not DSN:
    pytest.skip("LATTICE_TEST_PG_DSN not set", allow_module_level=True)

SCHEMA = Path(__file__).parent.parent.parent / "lattice" / "db" / "schema.sql"
DIM = 1024


def _vec(*nonzero: tuple[int, float]) -> list[float]:
    v = [0.0] * DIM
    for idx, val in nonzero:
        v[idx] = val
    return v


@pytest.fixture
async def pool():  # type: ignore[no-untyped-def]
    from lattice.db.vector import create_pg_pool

    pool = await create_pg_pool(DSN)
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA.read_text())
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM papers")
        await conn.execute(
            "INSERT INTO papers (paper_id, workspace_id, title) VALUES "
            "('p1','ws','Copper LSTM'), ('p2','ws','Copper VAR') "
            "ON CONFLICT (paper_id) DO NOTHING"
        )
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM papers")
    await pool.close()


async def test_upsert_and_ann_search(pool) -> None:  # type: ignore[no-untyped-def]
    from lattice.db.vector import PgVectorStore, VectorRecord

    store = PgVectorStore(pool=pool, workspace_id="ws")
    await store.upsert_chunks(
        [
            VectorRecord("c1", "p1", "ws", "Copper LSTM", "Results",
                         "LSTM beats ARIMA on RMSE for copper", _vec((0, 1.0)), "Table 1"),
            VectorRecord("c2", "p2", "ws", "Copper VAR", "Results",
                         "VAR underperforms on commodity panels", _vec((1, 1.0)), "Sec 4"),
        ]
    )
    # Query closest to the first vector -> p1 should rank first.
    ranked = await store.ann_search_papers(_vec((0, 1.0)), k=2, workspace_id="ws")
    assert ranked[0][0] == "p1"
    assert ranked[0][1] > ranked[1][1]


async def test_hybrid_search_combines_vector_and_text(pool) -> None:  # type: ignore[no-untyped-def]
    from lattice.db.vector import PgVectorStore, VectorRecord

    store = PgVectorStore(pool=pool, workspace_id="ws")
    await store.upsert_chunks(
        [
            VectorRecord("c1", "p1", "ws", "Copper LSTM", "Results",
                         "LSTM attention beats ARIMA on RMSE", _vec((0, 1.0)), "Table 1"),
            VectorRecord("c2", "p2", "ws", "Copper VAR", "Methods",
                         "vector autoregression on commodity panels", _vec((5, 1.0)), "Sec 2"),
        ]
    )
    hits = await store.hybrid_search("ARIMA RMSE", _vec((0, 1.0)), k=2, filters={"workspace_id": "ws"})
    assert hits and hits[0].paper_id == "p1"
    assert hits[0].evidence_location == "Table 1"


async def test_upsert_is_idempotent(pool) -> None:  # type: ignore[no-untyped-def]
    from lattice.db.vector import PgVectorStore, VectorRecord

    store = PgVectorStore(pool=pool, workspace_id="ws")
    rec = VectorRecord("c1", "p1", "ws", "T", "S", "text one", _vec((0, 1.0)), "L")
    await store.upsert_chunks([rec])
    rec2 = VectorRecord("c1", "p1", "ws", "T", "S", "text two", _vec((0, 1.0)), "L")
    await store.upsert_chunks([rec2])  # same chunk_id -> update, not duplicate
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM chunks WHERE chunk_id = 'c1'")
        text = await conn.fetchval("SELECT text FROM chunks WHERE chunk_id = 'c1'")
    assert count == 1
    assert text == "text two"
