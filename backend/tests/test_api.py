from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from lattice.api.app import create_app
from lattice.api.deps import Container, set_container
from lattice.config import Settings
from lattice.core.llm import LLMMessage, LLMResponse
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
    async def process_fulltext(self, pdf_bytes: bytes, filename: str = "paper.pdf") -> ParsedDocument:
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
            "methodology": {"approach_summary": "LSTM", "techniques": ["LSTM"], "baselines": ["ARIMA"]},
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
    llm = ScriptedLLM([_card_json("a"), _card_json("b")] + ['{"answer": "x", "citations": [], "confidence": 0.0}'] * 5)
    vectors = InMemoryVectorStore()
    cards = InMemoryCardStore()
    ingestion = IngestionService(
        settings=settings,
        llm=llm,
        parser=FakeParser(),
        vectors=vectors,
        cards=cards,
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


def test_metrics_endpoint_records_requests(client: TestClient) -> None:
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "lattice_http_requests_total" in r.text
    assert "lattice_http_request_duration_seconds_bucket" in r.text


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


def test_ingest_bad_pdf_fails_gracefully(client: TestClient) -> None:
    # Stage failures are captured as job state (resumable), not HTTP errors.
    r = client.post("/ingest/file", files={"file": ("bad.pdf", b"nope", "application/pdf")})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "failed"
    assert job["error_code"] == "corrupted_pdf"


def test_graph_and_jobs(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    client.post("/ingest/file", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})
    graph = client.get("/graph").json()
    assert len(graph["nodes"]) == 2
    stats = client.get("/graph/stats").json()
    assert stats["papers"] == 2
    jobs = client.get("/ingest/jobs").json()
    assert len(jobs) == 2


def test_landscape_matrix(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    r = client.get("/landscape/matrix?row=method&col=dataset")
    assert r.status_code == 200
    body = r.json()
    assert body["row_facet"] == "method"
    assert any(cell["row"] == "lstm" for cell in body["cells"])


def test_query_non_stream(client: TestClient) -> None:
    client.post("/ingest/file", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    r = client.post("/query", json={"question": "What is unknown?"})
    assert r.status_code == 200
    assert "answer" in r.json()


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
        assert c.get("/papers", headers={"Authorization": "Bearer secret"}).status_code in (200, 500)
    finally:
        monkeypatch.delenv("LATTICE_AUTH_TOKEN", raising=False)
        get_settings.cache_clear()
