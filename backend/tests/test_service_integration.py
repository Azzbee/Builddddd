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
            ParsedSection(
                section_id="s1",
                title="Methodology",
                text=f"We train {methods} on daily LME copper prices.",
            ),
            ParsedSection(
                section_id="s2",
                title="Results",
                text="Our model reaches RMSE 0.12, beating ARIMA at 0.21.",
            ),
        ],
        references=[ParsedReference(raw=r, doi=r) for r in refs],
    )


class FakeParser:
    def __init__(self, docs: dict[str, ParsedDocument]) -> None:
        self.docs = docs

    async def process_fulltext(
        self, pdf_bytes: bytes, filename: str = "paper.pdf"
    ) -> ParsedDocument:
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
                {
                    "claim": "beats ARIMA",
                    "metric": "RMSE",
                    "value": "0.12",
                    "evidence_location": "Results",
                }
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
        return LLMResponse(
            text=self._responses.pop(0), input_tokens=800, output_tokens=300, model=model
        )


def _service(
    parser: FakeParser,
    llm: ScriptedLLM,
    graph: FakeGraphStore,
    *,
    vectors: InMemoryVectorStore | None = None,
    cards: InMemoryCardStore | None = None,
) -> IngestionService:
    return IngestionService(
        settings=Settings(),
        llm=llm,
        parser=parser,
        vectors=vectors or InMemoryVectorStore(),
        cards=cards or InMemoryCardStore(),
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
        "RMSE copper",
        svc.chunk_embedder.embed_texts(["RMSE copper"])[0],
        5,
        {"workspace_id": "default"},
    )
    assert hits and any("RMSE" in h.text for h in hits)

    # Paper nodes + a RELATED_TO edge between the two similar papers were written.
    assert any("MERGE (p:Paper" in q for q, _ in graph.calls)
    assert any("RELATED_TO" in q for q, _ in graph.calls)


async def test_linking_survives_a_process_restart() -> None:
    # A fresh IngestionService that shares the persistent stores but starts with an
    # empty in-memory pool (simulating a process restart) must still link a new paper
    # against papers ingested before the "restart", and must reuse their Method node
    # instead of minting a duplicate. This exercises hydrate().
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()

    # Process 1: ingest one LSTM paper.
    docs1 = {"a.pdf": _doc("LSTM copper forecasting", "LSTM attention", ["10.1/shared", "10.1/a"])}
    svc1 = _service(
        FakeParser(docs1),
        ScriptedLLM([_content(["LSTM", "attention"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    await svc1.ingest_pdf("a.pdf", _pdf("A"))
    assert len(await shared_cards.all_cards()) == 1
    # The SPECTER vector was persisted, so it can be rehydrated.
    feats = await shared_cards.load_features()
    assert (
        feats and feats[0].specter is not None and "lstm" in {m.lower() for m in feats[0].methods}
    )

    # Process 2: brand-new service, same stores, empty in-memory linking state.
    graph2 = FakeGraphStore()
    docs2 = {"b.pdf": _doc("LSTM copper prediction", "LSTM gru", ["10.1/shared", "10.1/b"])}
    svc2 = _service(
        FakeParser(docs2),
        ScriptedLLM([_content(["LSTM", "GRU"])]),
        graph2,
        vectors=shared_vectors,
        cards=shared_cards,
    )
    assert not svc2._features  # fresh process: nothing loaded yet
    job_b = await svc2.ingest_pdf("b.pdf", _pdf("B"))
    assert job_b.status == JobStatus.SUCCEEDED

    # hydrate() pulled paper A into the candidate pool, so B linked against it.
    assert job_b.paper_id in svc2._features
    assert len(svc2._features) == 2, "the pre-restart paper must be in the candidate pool"
    assert any("RELATED_TO" in q for q, _ in graph2.calls), "new paper linked to the old one"

    # The shared "LSTM" method resolves to ONE canonical key across both processes
    # (no duplicate Method node from the restart).
    key1 = svc1.method_resolver.resolve("LSTM").key  # type: ignore[union-attr]
    key2 = svc2.method_resolver.resolve("LSTM").key  # type: ignore[union-attr]
    assert key1 == key2


async def test_hydrate_is_idempotent_and_keeps_richer_in_memory_features() -> None:
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    docs = {"a.pdf": _doc("LSTM copper forecasting", "LSTM attention", ["10.1/a"])}
    svc = _service(
        FakeParser(docs),
        ScriptedLLM([_content(["LSTM", "attention"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    job = await svc.ingest_pdf("a.pdf", _pdf("A"))
    pid = job.paper_id
    assert pid is not None
    # The just-ingested paper carries a methodology embedding in memory.
    before = svc._features[pid].methodology_embedding
    assert before is not None
    # Calling hydrate again must not overwrite the richer in-memory feature.
    svc._hydrated = False
    await svc.hydrate()
    assert svc._features[pid].methodology_embedding is before
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


def _doc_year(title: str, methods: str, year: int) -> ParsedDocument:
    d = _doc(title, methods, ["10.1/shared", f"10.1/{title}"])
    d.year = year
    return d


async def test_time_travel_snapshot_timeline_and_delta() -> None:
    docs = {
        "old.pdf": _doc_year("LSTM copper forecasting", "LSTM attention", 2018),
        "mid.pdf": _doc_year("GRU copper forecasting", "GRU LSTM", 2020),
        "new.pdf": _doc_year("Transformer copper forecasting", "transformer attention", 2023),
    }
    llm = ScriptedLLM(
        [_content(["LSTM", "attention"]), _content(["GRU"]), _content(["transformer"])]
    )
    svc = _service(FakeParser(docs), llm, FakeGraphStore())
    await svc.ingest_pdf("old.pdf", _pdf("O"))
    await svc.ingest_pdf("mid.pdf", _pdf("M"))
    await svc.ingest_pdf("new.pdf", _pdf("N"))

    # As of 2019 only the 2018 paper exists.
    snap_2019 = await svc.graph_snapshot(as_of_year=2019)
    assert {n.year for n in snap_2019.nodes} == {2018}

    # As of 2021 the 2018 + 2020 papers exist; never the 2023 one.
    snap_2021 = await svc.graph_snapshot(as_of_year=2021)
    assert {n.year for n in snap_2021.nodes} == {2018, 2020}
    assert all(n.year <= 2021 for n in snap_2021.nodes)

    # "Now" includes everything.
    assert len((await svc.graph_snapshot()).nodes) == 3

    # Timeline reports bounds and cumulative growth.
    tl = await svc.graph_timeline()
    assert tl["min_year"] == 2018 and tl["max_year"] == 2023
    last = tl["buckets"][-1]
    assert last["year"] == 2023 and last["papers"] == 3

    # Delta from 2019 -> now surfaces the two later papers.
    d = await svc.graph_delta(2019)
    assert d["counts"]["papers"] == 2
    assert {p["year"] for p in d["new_papers"]} == {2020, 2023}


async def test_research_proposal_and_opportunities() -> None:
    # lstm used on copper; var used on a different dataset -> lstm x var-dataset is a gap.
    docs = {
        "a.pdf": _doc_year("LSTM copper", "LSTM", 2021),
        "b.pdf": _doc_year("VAR panels", "VAR", 2020),
    }
    # Distinct datasets so a real empty cell (lstm x comex gold) exists.
    llm = ScriptedLLM(
        [
            _content(["LSTM"]),  # dataset: LME Copper
            _content(["VAR"]).replace("LME Copper", "COMEX Gold"),
        ]
    )
    svc = _service(FakeParser(docs), llm, FakeGraphStore())
    await svc.ingest_pdf("a.pdf", _pdf("A"))
    await svc.ingest_pdf("b.pdf", _pdf("B"))

    prop = await svc.research_proposal("method", "dataset", "lstm", "comex gold")
    assert prop["row"] == "lstm" and prop["col"] == "comex gold"
    assert prop["state"] in ("gap", "blind_spot")
    assert "lstm" in prop["thesis"].lower()
    assert "markdown" in prop and prop["markdown"].startswith("# Research proposal")

    opps = await svc.research_opportunities("method", "dataset", limit=3)
    assert opps["row_facet"] == "method"
    assert isinstance(opps["proposals"], list)


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
