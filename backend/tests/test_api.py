from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from lattice.api.app import create_app
from lattice.api.deps import Container, set_container
from lattice.config import Settings
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.db.aux_stores import InMemoryDigestStore, InMemoryWatchStore
from lattice.db.blobs import InMemoryBlobStore
from lattice.db.cards import InMemoryCardStore, InMemoryJobStore
from lattice.db.vector import InMemoryVectorStore
from lattice.embeddings.chunks import ChunkEmbedder
from lattice.embeddings.specter2 import Specter2Embedder
from lattice.graph.store import FakeGraphStore
from lattice.ingestion.models import ParsedDocument, ParsedSection
from lattice.ingestion.service import IngestionService

pytest.importorskip("lxml")


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        text = self._responses.pop(0) if self._responses else "{}"
        return LLMResponse(text=text, input_tokens=100, output_tokens=50, model=model)


class FakeParser:
    async def process_fulltext(
        self, pdf_bytes: bytes, filename: str = "paper.pdf"
    ) -> ParsedDocument:
        return ParsedDocument(
            title=f"Paper {filename}",
            authors=["Jane Doe"],
            abstract="We forecast copper prices with an LSTM and beat ARIMA.",
            year=2024,
            sections=[
                ParsedSection(section_id="s1", title="Results", text="RMSE 0.12 beats ARIMA 0.21."),
            ],
        )


def _card_json(title_tag: str) -> str:
    return json.dumps(
        {
            "problem_statement": "Forecasting copper prices is hard.",
            "research_questions": [],
            "methodology": {
                "approach_summary": "LSTM",
                "techniques": ["LSTM"],
                "baselines": ["ARIMA"],
            },
            "datasets": [{"name": "LME Copper"}],
            "key_results": [{"claim": "beats ARIMA", "evidence_location": "Results"}],
            "limitations": ["single commodity"],
            "contributions": ["model"],
            "future_work": [],
            "paper_type": "empirical",
            "domains": ["commodity markets"],
            "methods_taxonomy": ["LSTM"],
            "self_confidence": 0.9,
        }
    )


@pytest.fixture
def client() -> TestClient:
    settings = Settings()
    llm = ScriptedLLM(
        [_card_json("a"), _card_json("b")]
        + ['{"answer": "x", "citations": [], "confidence": 0.0}'] * 5
    )
    vectors = InMemoryVectorStore()
    cards = InMemoryCardStore()
    blobs = InMemoryBlobStore()
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=FakeParser(),
        vectors=vectors,
        cards=cards,
        blobs=blobs,
        graph=FakeGraphStore(),
        specter=Specter2Embedder(dim=64),
        chunk_embedder=ChunkEmbedder(dim=64),
        text_extractor=lambda b: "Real paper text. " * 60,
    )
    container = Container(
        settings=settings,
        llm=llm,
        vectors=vectors,
        cards=cards,
        jobs=InMemoryJobStore(),
        graph=ingestion.graph,
        chunk_embedder=ingestion.chunk_embedder,
        ingestion=ingestion,
        watch=InMemoryWatchStore(),
        digests=InMemoryDigestStore(),
        blobs=blobs,
    )
    set_container(container)
    yield TestClient(create_app())
    set_container(None)


def _pdf(tag: str) -> bytes:
    return b"%PDF-1.7\n" + tag.encode() + b" " + b"x" * 5000


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_reports_dependency_checks(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgres": True, "neo4j": True, "redis": True},
    }


def test_readiness_returns_503_when_a_dependency_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_check() -> dict[str, bool]:
        return {"postgres": True, "neo4j": False, "redis": True}

    monkeypatch.setattr("lattice.api.health.persistence_health", failed_check)
    response = TestClient(create_app()).get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_rate_limiting_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_RATE_LIMIT_PER_MIN", "3")
    get_settings.cache_clear()
    try:
        c = TestClient(create_app())
        # /metrics is exempt; /papers is not.
        codes = [c.get("/papers").status_code for _ in range(6)]
        assert 429 in codes
        assert codes.count(200) <= 3
        assert c.get("/health").status_code == 200  # exempt path always allowed
    finally:
        get_settings.cache_clear()


def test_rate_limit_cannot_be_rotated_with_unverified_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_RATE_LIMIT_PER_MIN", "2")
    monkeypatch.delenv("LATTICE_AUTH_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        c = TestClient(create_app())
        codes = [
            c.get("/papers", headers={"Authorization": f"Bearer invented-{index}"}).status_code
            for index in range(5)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2:] == [429, 429, 429]
    finally:
        get_settings.cache_clear()


def test_metrics_endpoint_records_requests(client: TestClient) -> None:
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "lattice_http_requests_total" in r.text
    assert "lattice_http_request_duration_seconds_bucket" in r.text


def test_papers_list_carries_superseded_marker(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    items = client.get("/papers").json()
    assert items and "superseded_by" in items[0] and items[0]["superseded_by"] is None


def test_ingest_and_read_paper(client: TestClient) -> None:
    r = client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "succeeded"
    pid = job["paper_id"]

    # List + fetch the card.
    papers = client.get("/papers").json()
    assert len(papers) == 1
    card = client.get(f"/papers/{pid}").json()
    assert card["title"].startswith("Paper")
    assert card["key_results"][0]["evidence_location"] == "Results"


@respx.mock
def test_arxiv_ingest_streams_a_valid_pdf(client: TestClient) -> None:
    respx.get("https://arxiv.org/pdf/2401.01234.pdf").mock(
        return_value=httpx.Response(200, content=_pdf("arxiv"))
    )

    response = client.post("/ingest/arxiv", json={"arxiv_id": "2401.01234"})

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


@respx.mock
def test_arxiv_ingest_rejects_oversized_stream(client: TestClient) -> None:
    from lattice.api.deps import get_container

    container = get_container("default")
    container.settings = container.settings.model_copy(update={"max_upload_mb": 1})
    respx.get("https://arxiv.org/pdf/2401.01234.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.7\n" + b"x" * (1024 * 1024))
    )

    response = client.post("/ingest/arxiv", json={"arxiv_id": "2401.01234"})

    assert response.status_code == 413
    assert response.json() == {"detail": "remote PDF is too large"}
    assert client.get("/ingest/jobs").json() == []


@respx.mock
def test_arxiv_ingest_rejects_oversized_content_length_before_read(
    client: TestClient,
) -> None:
    from lattice.api.deps import get_container

    container = get_container("default")
    container.settings = container.settings.model_copy(update={"max_upload_mb": 1})
    respx.get("https://arxiv.org/pdf/2401.01234.pdf").mock(
        return_value=httpx.Response(
            200,
            headers={"content-length": str(2 * 1024 * 1024)},
            content=b"%PDF-1.7\nsmall",
        )
    )

    response = client.post("/ingest/arxiv", json={"arxiv_id": "2401.01234"})

    assert response.status_code == 413
    assert response.json() == {"detail": "remote PDF is too large"}


def test_pdf_storage_and_streaming(client: TestClient) -> None:
    r = client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    pid = r.json()["paper_id"]

    meta = client.get(f"/papers/{pid}/pdf/meta").json()
    assert meta["available"] is True
    assert meta["size"] > 0

    pdf = client.get(f"/papers/{pid}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")

    # Unknown paper -> no PDF.
    assert client.get("/papers/missing/pdf/meta").json()["available"] is False
    assert client.get("/papers/missing/pdf").status_code == 404


def test_summarize_selection(client: TestClient) -> None:
    a = client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")}).json()
    b = client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")}).json()
    r = client.post("/papers/summarize", json={"paper_ids": [a["paper_id"], b["paper_id"]]})
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["count"] == 2
    assert "# Selection brief" in summary["markdown"]
    assert isinstance(summary["methods"], list)


def test_ingest_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    from lattice.api import deps
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_MAX_UPLOAD_MB", "0")  # reject anything non-empty
    get_settings.cache_clear()
    deps._registry.clear()  # drop any cached container so new settings take effect
    try:
        c = TestClient(create_app())
        r = c.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
        assert r.status_code == 413
    finally:
        get_settings.cache_clear()
        deps._registry.clear()


def test_ingest_bad_pdf_fails_gracefully(client: TestClient) -> None:
    # Stage failures are captured as job state (resumable), not HTTP errors.
    r = client.post("/ingest/file", files={"file": ("bad.pdf", b"nope", "application/pdf")})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "failed"
    assert job["error_code"] == "corrupted_pdf"
    assert job["retryable"] is False
    retry = client.post(f"/ingest/jobs/{job['job_id']}/retry")
    assert retry.status_code == 409
    assert retry.json()["detail"] == "job failure corrupted_pdf is not retryable"


def test_ingest_returns_503_when_queue_is_unavailable(client: TestClient) -> None:
    from lattice.api import deps
    from lattice.ingestion.dispatch import JobQueueUnavailable
    from lattice.ingestion.models import IngestJob

    class UnavailableDispatcher:
        asynchronous = True

        async def submit(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
            raise JobQueueUnavailable("Redis did not accept the ingestion job")

        async def retry(self, job_id: str) -> IngestJob | None:
            raise JobQueueUnavailable("Redis did not accept the ingestion job")

    assert deps._override is not None
    deps._override.dispatcher = UnavailableDispatcher()

    upload = client.post(
        "/ingest/file",
        files={"file": ("paper.pdf", _pdf("A"), "application/pdf")},
    )
    retry = client.post("/ingest/jobs/job-id/retry")
    assert upload.status_code == 503
    assert retry.status_code == 503
    assert upload.json()["detail"] == "Redis did not accept the ingestion job"


def test_graph_and_jobs(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    graph = client.get("/graph").json()
    assert len(graph["nodes"]) == 2
    stats = client.get("/graph/stats").json()
    assert stats["papers"] == 2
    jobs = client.get("/ingest/jobs").json()
    assert len(jobs) == 2


def test_graph_time_travel_endpoints(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    # The fake parser stamps year 2024 on every paper.
    tl = client.get("/graph/timeline").json()
    assert tl["min_year"] == 2024 and tl["max_year"] == 2024
    assert tl["buckets"][-1]["papers"] == 2

    # Before 2024 the graph is empty; at 2024 both papers appear.
    assert len(client.get("/graph?as_of_year=2023").json()["nodes"]) == 0
    assert len(client.get("/graph?as_of_year=2024").json()["nodes"]) == 2

    delta = client.get("/graph/delta?since_year=2023").json()
    assert delta["counts"]["papers"] == 2


def test_landscape_matrix(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    r = client.get("/landscape/matrix?row=method&col=dataset")
    assert r.status_code == 200
    body = r.json()
    assert body["row_facet"] == "method"
    assert any(cell["row"] == "lstm" for cell in body["cells"])


def test_landscape_proposal_and_opportunities(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    # use_global=false keeps it offline (no OpenAlex).
    r = client.get("/landscape/proposal?row=lstm&col=comex%20gold&use_global=false")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row"] == "lstm" and body["col"] == "comex gold"
    assert body["markdown"].startswith("# Research proposal")

    opp = client.get("/landscape/opportunities?row_facet=method&col_facet=dataset&use_global=false")
    assert opp.status_code == 200
    assert "proposals" in opp.json()


async def test_global_signal_is_time_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # A slow/hung OpenAlex must never block a landscape request: the helper degrades
    # to no global signal once the budget elapses.
    import asyncio

    from lattice.api import landscape
    from lattice.config import Settings

    async def _hang(*_a: object, **_k: object) -> dict:
        await asyncio.sleep(5.0)
        return {("x", "y"): 99}

    monkeypatch.setattr(landscape, "fetch_global_counts", _hang)
    monkeypatch.setattr(landscape, "GLOBAL_SIGNAL_BUDGET_S", 0.1)
    from lattice.api.deps import build_container

    c = build_container(Settings())
    loop = asyncio.get_event_loop()
    start = loop.time()
    result = await landscape._global_counts(c, [("x", "y")])
    assert result == {}  # degraded, not the hung value
    assert loop.time() - start < 2.0  # returned promptly, did not wait 5s


def test_query_non_stream(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    r = client.post("/query", json={"question": "What is unknown?"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_query_non_stream_survives_top_level_non_object_json() -> None:
    # The non-streaming /query has no outer try/except, so a top-level non-object
    # from the LLM (valid JSON, but an array) must be handled inside the agent loop
    # rather than 500ing the endpoint.
    settings = Settings()
    # First reply is a bare array (invalid shape), then a proper final answer.
    llm = ScriptedLLM(["[1, 2, 3]", '{"answer": "recovered", "citations": [], "confidence": 0.3}'])
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=FakeParser(),
        vectors=InMemoryVectorStore(),
        cards=InMemoryCardStore(),
        blobs=InMemoryBlobStore(),
        graph=FakeGraphStore(),
        specter=Specter2Embedder(dim=64),
        chunk_embedder=ChunkEmbedder(dim=64),
        text_extractor=lambda b: "Real paper text. " * 60,
    )
    container = Container(
        settings=settings,
        llm=llm,
        vectors=ingestion.vectors,
        cards=ingestion.cards,
        jobs=InMemoryJobStore(),
        graph=ingestion.graph,
        chunk_embedder=ingestion.chunk_embedder,
        ingestion=ingestion,
        watch=InMemoryWatchStore(),
        digests=InMemoryDigestStore(),
        blobs=ingestion.blobs,
    )
    set_container(container)
    try:
        r = TestClient(create_app()).post("/query", json={"question": "anything"})
    finally:
        set_container(None)
    assert r.status_code == 200
    assert r.json()["answer"] == "recovered"


class RaisingLLM:
    async def complete(self, model, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider is down")


def test_query_stream_ends_with_final_when_agent_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the agent raises mid-stream (provider outage), the SSE stream must still
    # emit exactly one terminal `final` event so the client UI recovers instead of
    # hanging forever on "thinking...".
    settings = Settings()
    llm = RaisingLLM()
    vectors = InMemoryVectorStore()
    cards = InMemoryCardStore()
    blobs = InMemoryBlobStore()
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=FakeParser(),
        vectors=vectors,
        cards=cards,
        blobs=blobs,
        graph=FakeGraphStore(),
        specter=Specter2Embedder(dim=64),
        chunk_embedder=ChunkEmbedder(dim=64),
        text_extractor=lambda b: "Real paper text. " * 60,
    )
    container = Container(
        settings=settings,
        llm=llm,
        vectors=vectors,
        cards=cards,
        jobs=InMemoryJobStore(),
        graph=ingestion.graph,
        chunk_embedder=ingestion.chunk_embedder,
        ingestion=ingestion,
        watch=InMemoryWatchStore(),
        digests=InMemoryDigestStore(),
        blobs=blobs,
    )
    set_container(container)
    try:
        c = TestClient(create_app())
        with c.stream("POST", "/query/stream", json={"question": "anything"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
    finally:
        set_container(None)

    # Parse SSE frames (CRLF-delimited): the last event must be `final` + error.
    frames = [f for f in body.replace("\r\n", "\n").split("\n\n") if f.strip()]
    assert frames, "stream produced no events"
    last = frames[-1]
    assert "event: final" in last
    data_line = next(line for line in last.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line[len("data:") :].strip())
    assert payload["error"] is True
    assert payload["abstained"] is True


def test_landscape_quadrants(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    r = client.get("/landscape/quadrants")
    assert r.status_code == 200
    body = r.json()
    assert "known_knowns" in body and "known_unknowns" in body and "unknown_knowns" in body


def test_landscape_matrix_local_only(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    # use_global=false avoids any network; still returns a valid matrix.
    r = client.get("/landscape/matrix?row=method&col=dataset&use_global=false")
    assert r.status_code == 200
    assert r.json()["global_signal"] is False


def test_related_work_and_exports(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    rw = client.get("/related-work").json()
    assert "markdown" in rw and "# Related work" in rw["markdown"]
    bib = client.get("/export/bibtex")
    assert bib.status_code == 200 and "@" in bib.text
    obs = client.get("/export/obsidian")
    assert obs.status_code == 200
    assert obs.headers["content-type"] == "application/zip"
    assert len(obs.content) > 0


def test_digest_generate_and_latest(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    gen = client.post("/digest/generate").json()
    assert "markdown" in gen and "# Lattice digest" in gen["markdown"]
    latest = client.get("/digest/latest").json()
    assert latest["digest"] is not None


def test_lineage(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    r = client.get("/lineage?method=lstm")
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "lstm" and "nodes" in body and "timeline" in body


def test_watch_queue_and_approve(client: TestClient) -> None:
    from lattice.api.deps import get_container

    # The override container uses the in-memory watch store; seed it directly.
    store = get_container("default").watch
    store._items["2406.123"] = {  # type: ignore[attr-defined]
        "arxiv_id": "2406.123",
        "title": "Cand",
        "similarity": 0.6,
        "status": "pending",
    }
    queue = client.get("/watch/queue").json()
    assert any(item["arxiv_id"] == "2406.123" for item in queue)
    r = client.post("/watch/approve", json={"arxiv_id": "2406.123", "approve": True})
    assert r.json()["status"] == "approved"
    # Approved item leaves the pending queue.
    assert all(item["arxiv_id"] != "2406.123" for item in client.get("/watch/queue").json())


def test_reading_queue(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    r = client.get("/reading-queue")
    assert r.status_code == 200
    body = r.json()
    assert "queue" in body and isinstance(body["queue"], list)


def test_contradictions_analyze(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    r = client.post("/contradictions/analyze")
    assert r.status_code == 200
    body = r.json()
    assert "analyzed" in body and "supports" in body
    # Both papers claim "beats ARIMA" on the same concept -> a SUPPORTS relation.
    assert body["supports"] >= 1
    listed = client.get("/contradictions?relation=SUPPORTS").json()
    assert len(listed) >= 1


def test_auth_enforced_when_token_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_AUTH_TOKEN", "secret")
    get_settings.cache_clear()
    try:
        app = create_app()
        c = TestClient(app)
        assert c.get("/papers").status_code == 401
        assert c.get("/metrics").status_code == 401
        assert c.get("/workspaces").status_code == 401
        assert c.get("/papers", headers={"Authorization": "Bearer secret"}).status_code in (
            200,
            500,
        )
        assert c.get("/workspaces", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code == 200
    finally:
        monkeypatch.delenv("LATTICE_AUTH_TOKEN", raising=False)
        get_settings.cache_clear()


def test_production_disables_interactive_api_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from lattice.config import get_settings

    monkeypatch.setenv("LATTICE_ENVIRONMENT", "prod")
    monkeypatch.setenv("LATTICE_AUTH_TOKEN", "secret")
    get_settings.cache_clear()
    try:
        c = TestClient(create_app())
        assert c.get("/docs").status_code == 404
        assert c.get("/redoc").status_code == 404
        assert c.get("/openapi.json").status_code == 404
    finally:
        get_settings.cache_clear()
