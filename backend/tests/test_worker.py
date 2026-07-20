from __future__ import annotations

import pytest
from lattice.api.deps import build_container
from lattice.config import Settings
from lattice.demo import load_demo
from lattice.enrichment.models import ArxivResult

pytest.importorskip("lxml")


@pytest.fixture(autouse=True)
def _low_watch_floor(monkeypatch: pytest.MonkeyPatch):
    # The offline hashing embedder yields modest cosine; lower the watcher floor so
    # an on-topic candidate clears it deterministically. poll_arxiv reads global settings.
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_WATCHER__SIMILARITY_FLOOR", "0.05")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _demo_ctx() -> dict[str, object]:
    settings = Settings(demo_mode=True)
    container = build_container(settings)
    await load_demo(container)
    return {
        "container": container,
        "settings": settings,
        "containers": {settings.workspace_id: container},
    }


class FakeWatcher:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def fetch_recent(self, categories, max_results=50):  # type: ignore[no-untyped-def]
        return [
            ArxivResult(
                arxiv_id="2406.99999",
                title="LSTM attention for copper price forecasting",
                summary="forecasting commodity prices with attention and lstm models",
            ),
            ArxivResult(
                arxiv_id="2406.00001",
                title="Quantum chromodynamics lattice gauge theory",
                summary="particle physics simulation unrelated to commodities",
            ),
        ]


async def test_poll_arxiv_queues_similar_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    import lattice.worker as worker

    monkeypatch.setattr(worker, "ArxivWatcher", FakeWatcher)
    ctx = await _demo_ctx()
    added = await worker.poll_arxiv(ctx)
    assert added >= 1
    queue = await ctx["container"].watch.pending()  # type: ignore[attr-defined]
    arxiv_ids = {item["arxiv_id"] for item in queue}
    assert "2406.99999" in arxiv_ids  # on-topic candidate queued


async def test_poll_arxiv_handles_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import lattice.worker as worker

    class Boom(FakeWatcher):
        async def fetch_recent(self, categories, max_results=50):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    monkeypatch.setattr(worker, "ArxivWatcher", Boom)
    ctx = await _demo_ctx()
    assert await worker.poll_arxiv(ctx) == 0  # degrades gracefully


async def test_generate_weekly_digest_persists() -> None:
    import lattice.worker as worker

    ctx = await _demo_ctx()
    payload = await worker.generate_weekly_digest(ctx)
    assert "markdown" in payload
    assert await ctx["container"].digests.latest() is not None  # type: ignore[attr-defined]


async def test_ingest_job_task_runs_from_staged_source() -> None:
    import lattice.worker as worker
    from lattice.demo import demo_pdf_bytes

    ctx = await _demo_ctx()
    container = ctx["container"]
    job = await container.ingestion.stage_pdf(  # type: ignore[attr-defined]
        "new-paper.pdf", demo_pdf_bytes("lstm_copper.pdf")
    )
    result = await worker.ingest_job_task(ctx, "default", job.job_id)
    assert result["status"] in ("succeeded", "duplicate")
