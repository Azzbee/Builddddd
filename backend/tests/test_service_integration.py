"""End-to-end ingestion: PDF bytes -> parse -> extract -> embed -> graph, offline.

Proves M1-M4 connect with in-memory stores, a scripted LLM, and a fake parser.
"""

from __future__ import annotations

import json

import pytest
from lattice.config import Settings
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.db.cards import InMemoryCardStore
from lattice.db.vector import InMemoryVectorStore
from lattice.embeddings.specter2 import Specter2Embedder
from lattice.graph.store import FakeGraphStore
from lattice.ingestion.models import (
    JobStatus,
    ParsedDocument,
    ParsedReference,
    ParsedSection,
)
from lattice.ingestion.service import IngestionService, assign_paper_id

pytest.importorskip("lxml")

PDF_HEADER = b"%PDF-1.7\n"


def _pdf(tag: str) -> bytes:
    return PDF_HEADER + tag.encode() + b" " + b"x" * 5000


def _doc(title: str, methods: str, refs: list[str]) -> ParsedDocument:
    return ParsedDocument(
        title=title,
        authors=["Jane Doe", "Carlos Ng"],
        abstract=f"We forecast copper prices using {methods}.",
        year=2024,
        doi=None,
        arxiv_id=None,
        sections=[
            ParsedSection(section_id="s1", title="Methodology",
                          text=f"We train {methods} on daily LME copper prices."),
            ParsedSection(section_id="s2", title="Results",
                          text="Our model reaches RMSE 0.12, beating ARIMA at 0.21."),
        ],
        references=[ParsedReference(raw=r, doi=r) for r in refs],
    )


class FakeParser:
    def __init__(self, docs: dict[str, ParsedDocument]) -> None:
        self.docs = docs

    async def process_fulltext(self, pdf_bytes: bytes, filename: str = "paper.pdf") -> ParsedDocument:
        return self.docs[filename]


def _content(methods: list[str]) -> str:
    return json.dumps(
        {
            "problem_statement": "Forecasting copper prices is hard and underexplored.",
            "research_questions": ["Can deep models beat ARIMA?"],
            "methodology": {
                "approach_summary": f"{methods[0]} based forecaster",
                "method_family": ["deep learning"],
                "techniques": methods,
                "baselines": ["ARIMA"],
                "reproducibility": {"code_available": True},
            },
            "datasets": [{"name": "LME Copper", "source": "LME", "is_public": True}],
            "key_results": [
                {"claim": "beats ARIMA", "metric": "RMSE", "value": "0.12", "evidence_location": "Results"}
            ],
            "limitations": ["single commodity"],
            "contributions": ["new model"],
            "future_work": ["other metals"],
            "paper_type": "empirical",
            "domains": ["commodity markets"],
            "methods_taxonomy": methods,
            "self_confidence": 0.9,
        }
    )


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(text=self._responses.pop(0), input_tokens=800, output_tokens=300, model=model)


def _service(parser: FakeParser, llm: ScriptedLLM, graph: FakeGraphStore) -> IngestionService:
    return IngestionService(
        settings=Settings(),
        llm=llm,
        parser=parser,
        vectors=InMemoryVectorStore(),
        cards=InMemoryCardStore(),
        graph=graph,
        specter=Specter2Embedder(dim=128),
        text_extractor=lambda b: "Real extracted paper text. " * 60,
    )


async def test_full_ingest_two_papers_creates_graph_and_edge() -> None:
    docs = {
        "paper_a.pdf": _doc("LSTM copper forecasting", "LSTM attention", ["10.1/shared", "10.1/a"]),
        "paper_b.pdf": _doc("LSTM copper prediction", "LSTM gru", ["10.1/shared", "10.1/b"]),
    }
    graph = FakeGraphStore()
    llm = ScriptedLLM([_content(["LSTM", "attention"]), _content(["LSTM", "GRU"])])
    svc = _service(FakeParser(docs), llm, graph)

    job_a = await svc.ingest_pdf("paper_a.pdf", _pdf("A"))
    job_b = await svc.ingest_pdf("paper_b.pdf", _pdf("B"))

    assert job_a.status == JobStatus.SUCCEEDED
    assert job_b.status == JobStatus.SUCCEEDED

    # Both cards stored.
    cards = await svc.cards.all_cards()
    assert len(cards) == 2

    # Chunks embedded and searchable.
    hits = await svc.vectors.hybrid_search(
        "RMSE copper", svc.chunk_embedder.embed_texts(["RMSE copper"])[0], 5,
        {"workspace_id": "default"},
    )
    assert hits and any("RMSE" in h.text for h in hits)

    # Paper nodes + a RELATED_TO edge between the two similar papers were written.
    assert any("MERGE (p:Paper" in q for q, _ in graph.calls)
    assert any("RELATED_TO" in q for q, _ in graph.calls)


async def test_source_pdf_is_stored_for_reader() -> None:
    docs = {"paper_a.pdf": _doc("LSTM copper forecasting", "LSTM", ["10.1/a"])}
    llm = ScriptedLLM([_content(["LSTM"])])
    svc = _service(FakeParser(docs), llm, FakeGraphStore())
    pdf = _pdf("A")
    job = await svc.ingest_pdf("paper_a.pdf", pdf)
    assert job.paper_id is not None
    # The raw PDF is persisted keyed by paper_id so the reader can open it.
    assert await svc.blobs.get(job.paper_id) == pdf
    meta = await svc.blobs.meta(job.paper_id)
    assert meta is not None and meta.size == len(pdf)


async def test_summarize_selected_papers_brief() -> None:
    docs = {
        "paper_a.pdf": _doc("LSTM copper forecasting", "LSTM attention", ["10.1/a"]),
        "paper_b.pdf": _doc("GRU copper forecasting", "GRU", ["10.1/b"]),
    }
    llm = ScriptedLLM([_content(["LSTM", "attention"]), _content(["GRU"])])
    svc = _service(FakeParser(docs), llm, FakeGraphStore())
    a = await svc.ingest_pdf("paper_a.pdf", _pdf("A"))
    b = await svc.ingest_pdf("paper_b.pdf", _pdf("B"))

    summary = await svc.summarize_papers([a.paper_id, b.paper_id])
    assert summary["count"] == 2
    assert summary["year_range"] == [2024, 2024]
    assert "LME Copper" in summary["datasets"]
    assert "commodity markets" in summary["domains"]
    assert "# Selection brief" in summary["markdown"]
    # Unknown ids are ignored; empty selection yields an empty brief.
    assert (await svc.summarize_papers(["does-not-exist"]))["count"] == 0


async def test_reingest_is_idempotent_duplicate() -> None:
    docs = {"paper_a.pdf": _doc("LSTM copper forecasting", "LSTM", ["10.1/a"])}
    graph = FakeGraphStore()
    llm = ScriptedLLM([_content(["LSTM"]), _content(["LSTM"])])
    svc = _service(FakeParser(docs), llm, graph)

    first = await svc.ingest_pdf("paper_a.pdf", _pdf("A"))
    assert first.status == JobStatus.SUCCEEDED

    # Re-ingest identical content -> deduped, not a second paper.
    second = await svc.ingest_pdf("paper_a.pdf", _pdf("A"))
    assert second.status == JobStatus.DUPLICATE
    assert len(await svc.cards.all_cards()) == 1


async def test_corrupted_pdf_fails_gracefully() -> None:
    docs = {"bad.pdf": _doc("x", "y", [])}
    svc = _service(FakeParser(docs), ScriptedLLM([]), FakeGraphStore())
    job = await svc.ingest_pdf("bad.pdf", b"not a pdf")
    assert job.status == JobStatus.FAILED
    assert job.error_code == "corrupted_pdf"


def test_assign_paper_id_prefers_doi() -> None:
    by_doi = assign_paper_id(ParsedDocument(title="t", doi="10.1/X"), "ws")
    by_doi2 = assign_paper_id(ParsedDocument(title="different", doi="https://doi.org/10.1/x"), "ws")
    assert by_doi == by_doi2  # same DOI -> same id regardless of title
