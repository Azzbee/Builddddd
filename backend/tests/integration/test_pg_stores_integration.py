"""Live PgCardStore + PgJobStore integration tests (gated on LATTICE_TEST_PG_DSN)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DSN = os.environ.get("LATTICE_TEST_PG_DSN")
if not DSN:
    pytest.skip("LATTICE_TEST_PG_DSN not set", allow_module_level=True)

from lattice.db.pg_stores import PgCardStore, PgJobStore  # noqa: E402
from lattice.extraction.schemas import Author, Methodology, PaperCard, PaperType  # noqa: E402
from lattice.ingestion.models import IngestJob, JobStage, JobStatus, SourceType  # noqa: E402

SCHEMA = Path(__file__).parent.parent.parent / "lattice" / "db" / "schema.sql"


@pytest.fixture
async def pool():  # type: ignore[no-untyped-def]
    from lattice.db.vector import create_pg_pool

    pool = await create_pg_pool(DSN)
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA.read_text())
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM ingest_jobs")
        await conn.execute("DELETE FROM papers")
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM ingest_jobs")
        await conn.execute("DELETE FROM papers")
    await pool.close()


def _card(pid: str, title: str) -> PaperCard:
    return PaperCard(
        paper_id=pid, title=title, authors=[Author(name="Jane Doe")], year=2024,
        doi=f"10.1/{pid}", methodology=Methodology(approach_summary="LSTM"),
        paper_type=PaperType.EMPIRICAL, methods_taxonomy=["LSTM"], confidence=0.9,
    )


async def test_card_roundtrip_and_corpus_index(pool) -> None:  # type: ignore[no-untyped-def]
    store = PgCardStore(pool, "ws")
    await store.put_card(_card("p1", "Copper LSTM"))
    await store.put_card(_card("p2", "Copper GRU"))

    got = await store.get_card("p1")
    assert got is not None and got["title"] == "Copper LSTM"
    full = await store.get("p1")
    assert full is not None and full.methods_taxonomy == ["LSTM"]
    assert len(await store.all_cards()) == 2

    index = await store.corpus_index()
    dup = index.find_duplicate(
        type(index.titles[0])(paper_id="x", title="Copper LSTM", authors=["Jane Doe"], doi="10.1/p1")
    )
    assert dup.is_duplicate and dup.existing_paper_id == "p1"


async def test_card_upsert_idempotent(pool) -> None:  # type: ignore[no-untyped-def]
    store = PgCardStore(pool, "ws")
    await store.put_card(_card("p1", "First"))
    await store.put_card(_card("p1", "Second"))  # same id -> update
    cards = await store.all_cards()
    assert len(cards) == 1 and cards[0].title == "Second"


async def test_job_roundtrip(pool) -> None:  # type: ignore[no-untyped-def]
    store = PgJobStore(pool, "ws")
    job = IngestJob(
        job_id="j1", workspace_id="ws", source_type=SourceType.FILE, source_ref="a.pdf",
        stage=JobStage.LINKING, status=JobStatus.SUCCEEDED, paper_id="p1", attempts=1, cost_usd=0.12,
    )
    await store.save(job)
    got = await store.get("j1")
    assert got is not None and got.status == JobStatus.SUCCEEDED and got.paper_id == "p1"
    assert got.stage == JobStage.LINKING
    assert len(await store.all_jobs()) == 1


async def test_job_workspace_isolation(pool) -> None:  # type: ignore[no-untyped-def]
    a = PgJobStore(pool, "wsA")
    b = PgJobStore(pool, "wsB")
    await a.save(IngestJob(job_id="j1", workspace_id="wsA", source_type=SourceType.FILE, source_ref="a"))
    assert len(await a.all_jobs()) == 1
    assert len(await b.all_jobs()) == 0
