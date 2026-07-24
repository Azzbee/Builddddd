"""End-to-end ingestion: PDF bytes -> parse -> extract -> embed -> graph, offline.

Proves M1-M4 connect with in-memory stores, a scripted LLM, and a fake parser.
"""

from __future__ import annotations

import json

import pytest
from lattice.config import Settings
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.db.cards import InMemoryCardStore
from lattice.db.ingest_artifacts import InMemoryIngestArtifactStore
from lattice.db.vector import InMemoryVectorStore
from lattice.embeddings.specter2 import Specter2Embedder
from lattice.graph.store import FakeGraphStore
from lattice.ingestion.dedup import PaperIdentity
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


def _doc_ids(
    title: str, methods: str, *, doi: str | None = None, arxiv: str | None = None
) -> ParsedDocument:
    d = _doc(title, methods, ["10.1/shared"])
    d.doi = doi
    d.arxiv_id = arxiv
    return d


class FakeParser:
    def __init__(self, docs: dict[str, ParsedDocument]) -> None:
        self.docs = docs
        self.calls = 0

    async def process_fulltext(
        self, pdf_bytes: bytes, filename: str = "paper.pdf"
    ) -> ParsedDocument:
        self.calls += 1
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


async def test_concurrent_hydrate_loads_exactly_once() -> None:
    # Two ingests firing concurrently (as arq workers would) must not each run the
    # feature load: the lock + double-check serialize hydration so load_features is
    # called exactly once. (Without any guard, both call it -> calls == 2.) This is
    # the deterministically-testable half of the fix; the lock additionally ensures a
    # late arrival waits for full population rather than racing a half-filled pool.
    import asyncio

    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    seed = _service(
        FakeParser({"s.pdf": _doc("LSTM seed", "LSTM", ["10.1/s"])}),
        ScriptedLLM([_content(["LSTM"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    await seed.ingest_pdf("s.pdf", _pdf("S"))
    seed_pid = (await shared_cards.all_cards())[0].paper_id

    calls = {"n": 0}
    orig = shared_cards.load_features

    async def slow_load():  # type: ignore[no-untyped-def]
        calls["n"] += 1
        await asyncio.sleep(0.05)  # hold the lock long enough for B to arrive
        return await orig()

    shared_cards.load_features = slow_load  # type: ignore[method-assign]

    svc = _service(
        FakeParser(
            {
                "a.pdf": _doc("GRU one", "GRU", ["10.1/a"]),
                "b.pdf": _doc("GRU two", "GRU", ["10.1/b"]),
            }
        ),
        ScriptedLLM([_content(["GRU"]), _content(["GRU"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    await asyncio.gather(svc.ingest_pdf("a.pdf", _pdf("A")), svc.ingest_pdf("b.pdf", _pdf("B")))
    assert calls["n"] == 1, "hydration must load exactly once despite concurrency"
    # The seeded paper is in the pool, so whichever ingest ran second still linked
    # against a fully populated corpus (not an empty one).
    assert seed_pid in svc._features


async def test_hydrate_retries_after_a_transient_store_failure() -> None:
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    seed = _service(
        FakeParser({"s.pdf": _doc("LSTM seed", "LSTM", ["10.1/s"])}),
        ScriptedLLM([_content(["LSTM"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    seeded_job = await seed.ingest_pdf("s.pdf", _pdf("S"))
    assert seeded_job.paper_id is not None

    original_load = shared_cards.load_features
    calls = 0

    async def flaky_load():  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("temporary database outage")
        return await original_load()

    shared_cards.load_features = flaky_load  # type: ignore[method-assign]
    restarted = _service(
        FakeParser({}),
        ScriptedLLM([]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )

    with pytest.raises(ConnectionError, match="temporary database outage"):
        await restarted.hydrate()
    assert restarted._hydrated is False
    assert restarted._features == {}

    await restarted.hydrate()

    assert calls == 2
    assert restarted._hydrated is True
    assert seeded_job.paper_id in restarted._features


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

    # Repeating the same deterministic job returns its existing terminal result.
    second = await svc.ingest_pdf("paper_a.pdf", _pdf("A"))
    assert second.status == JobStatus.SUCCEEDED
    assert second.job_id == first.job_id
    assert len(await svc.cards.all_cards()) == 1


async def test_failed_ingest_resumes_without_repeating_completed_parse() -> None:
    parser = FakeParser({"paper.pdf": _doc("LSTM copper", "LSTM", ["10.1/a"])})
    artifacts = InMemoryIngestArtifactStore()
    svc = _service(parser, ScriptedLLM(["not-json"] * 3), FakeGraphStore())
    svc.artifacts = artifacts

    failed = await svc.ingest_pdf("paper.pdf", _pdf("A"))
    assert failed.status == JobStatus.FAILED
    assert failed.stage.value == "parsing"
    assert parser.calls == 1
    saved = await artifacts.get(failed.job_id)
    assert saved is not None and saved.document is not None and saved.raw_pdf == _pdf("A")

    svc.llm = ScriptedLLM([_content(["LSTM"])])
    resumed = await svc.resume_job(failed.job_id)
    assert resumed is not None and resumed.status == JobStatus.SUCCEEDED
    assert parser.calls == 1


async def test_content_hash_dedup_survives_card_store_rehydration() -> None:
    cards = InMemoryCardStore()
    parser = FakeParser({"first.pdf": _doc("Original title", "LSTM", ["10.1/a"])})
    svc = _service(parser, ScriptedLLM([_content(["LSTM"])]), FakeGraphStore(), cards=cards)
    first = await svc.ingest_pdf("first.pdf", _pdf("SAME"))
    assert first.status == JobStatus.SUCCEEDED

    index = await cards.corpus_index()
    duplicate = index.find_duplicate(
        PaperIdentity("incoming", "Different metadata", content_hash=first.content_hash)
    )
    assert duplicate.is_duplicate and duplicate.reason == "content_hash"


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


# ----------------------------------------------------------------- supersession
async def test_published_version_supersedes_preprint() -> None:
    # The living-graph story: the arXiv preprint is ingested first; when the
    # published (DOI-bearing) version of the same manuscript arrives, it is NOT
    # rejected as a duplicate - it supersedes the preprint. Bi-temporal: the
    # preprint stays stored, its edges are invalidated, and it leaves the
    # candidate pool and analytics so the work is never counted twice.
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    graph = FakeGraphStore()
    docs = {
        "pre.pdf": _doc_ids("Copper forecasting with LSTM", "LSTM attention", arxiv="2401.00001"),
        "pub.pdf": _doc_ids(
            "Copper Forecasting with LSTM", "LSTM attention", doi="10.1234/jf.2024.7"
        ),
    }
    svc = _service(
        FakeParser(docs),
        ScriptedLLM([_content(["LSTM", "attention"]), _content(["LSTM", "attention"])]),
        graph,
        vectors=shared_vectors,
        cards=shared_cards,
    )
    pre = await svc.ingest_pdf("pre.pdf", _pdf("PRE"))
    assert pre.status == JobStatus.SUCCEEDED
    pub = await svc.ingest_pdf("pub.pdf", _pdf("PUB"))
    assert pub.status == JobStatus.SUCCEEDED, pub.error_message
    assert pre.paper_id != pub.paper_id  # two versions, two nodes

    # Store-level supersession flag.
    assert await shared_cards.superseded_map() == {pre.paper_id: pub.paper_id}
    # Graph-level: SUPERSEDED_BY edge written, preprint's edges invalidated.
    assert any("SUPERSEDED_BY" in q for q, _ in graph.calls)
    assert any("invalid_at = $now" in q and p.get("pid") == pre.paper_id for q, p in graph.calls)
    # The published version did NOT link to its own preprint.
    assert svc._related.get(pub.paper_id) == []
    # Candidate pool and mirrors: preprint gone.
    assert pre.paper_id not in svc._features
    assert pre.paper_id not in svc._related
    # Analytics and the default graph view count the work exactly once.
    assert [c.paper_id for c in await svc._active_cards()] == [pub.paper_id]
    snap = await svc.graph_snapshot()
    assert {n.id for n in snap.nodes} == {pub.paper_id}
    # But the preprint remains retrievable by id (supersede, never delete).
    assert await shared_cards.get(pre.paper_id) is not None


async def test_outdated_preprint_rejected_when_published_exists() -> None:
    shared_cards = InMemoryCardStore()
    docs = {
        "pub.pdf": _doc_ids("Copper Forecasting with LSTM", "LSTM", doi="10.1234/jf.2024.7"),
        "pre.pdf": _doc_ids("Copper forecasting with LSTM", "LSTM", arxiv="2401.00001"),
    }
    svc = _service(
        FakeParser(docs),
        ScriptedLLM([_content(["LSTM"])]),
        FakeGraphStore(),
        cards=shared_cards,
    )
    pub = await svc.ingest_pdf("pub.pdf", _pdf("PUB"))
    assert pub.status == JobStatus.SUCCEEDED
    pre = await svc.ingest_pdf("pre.pdf", _pdf("PRE"))
    assert pre.status == JobStatus.DUPLICATE
    assert "outdated preprint" in (pre.error_message or "")
    assert await shared_cards.superseded_map() == {}  # nothing was ingested


async def test_same_identifier_match_is_still_a_plain_duplicate() -> None:
    # Identifier-level matches (same DOI) are the same artifact, never supersession.
    docs = {
        "a.pdf": _doc_ids("Copper Forecasting with LSTM", "LSTM", doi="10.1/same"),
        "b.pdf": _doc_ids("Copper Forecasting with LSTM v2", "LSTM", doi="10.1/same"),
    }
    svc = _service(FakeParser(docs), ScriptedLLM([_content(["LSTM"])]), FakeGraphStore())
    assert (await svc.ingest_pdf("a.pdf", _pdf("A"))).status == JobStatus.SUCCEEDED
    dup = await svc.ingest_pdf("b.pdf", _pdf("B"))
    assert dup.status == JobStatus.DUPLICATE
    assert "duplicate of" in (dup.error_message or "")


async def test_superseded_paper_stays_out_of_pool_after_restart() -> None:
    # Hydration must not resurrect a superseded preprint into the candidate pool,
    # and rehydrated papers now carry their aspect vectors (full-fidelity restore).
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    docs1 = {
        "pre.pdf": _doc_ids("Copper forecasting with LSTM", "LSTM", arxiv="2401.00001"),
        "pub.pdf": _doc_ids("Copper Forecasting with LSTM", "LSTM", doi="10.1234/jf.2024.7"),
    }
    svc1 = _service(
        FakeParser(docs1),
        ScriptedLLM([_content(["LSTM"]), _content(["LSTM"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    pre = await svc1.ingest_pdf("pre.pdf", _pdf("PRE"))
    pub = await svc1.ingest_pdf("pub.pdf", _pdf("PUB"))

    # "Restart": fresh service, same stores. Ingest a third paper to trigger hydrate.
    docs2 = {"c.pdf": _doc_ids("GRU networks for metal prices", "GRU")}
    svc2 = _service(
        FakeParser(docs2),
        ScriptedLLM([_content(["GRU"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    third = await svc2.ingest_pdf("c.pdf", _pdf("C"))
    assert third.status == JobStatus.SUCCEEDED
    assert pre.paper_id not in svc2._features, "superseded preprint must not rehydrate"
    assert pub.paper_id in svc2._features
    # Full-fidelity hydration: the methodology aspect vector survived the restart.
    assert svc2._features[pub.paper_id].methodology_embedding is not None
    assert pub.paper_id in svc2._aspects


# --------------------------------------------------- incremental contradictions
def _content_with_claim(methods: list[str], claim: str) -> str:
    d = json.loads(_content(methods))
    d["key_results"] = [{"claim": claim, "evidence_location": "Results, Table 2"}]
    return json.dumps(d)


async def test_contradiction_surfaces_incrementally_on_ingest() -> None:
    # The living graph: paper B contradicts paper A, and the CONTRADICTS edge
    # exists right after B's ingest - no manual /contradictions/analyze pass.
    graph = FakeGraphStore()
    docs = {
        "a.pdf": _doc("LSTM improves copper forecasting", "LSTM", ["10.1/a"]),
        "b.pdf": _doc("A critical replication of neural commodity models", "LSTM", ["10.1/b"]),
    }
    llm = ScriptedLLM(
        [
            _content_with_claim(
                ["LSTM"], "LSTM significantly improves forecasting accuracy over ARIMA"
            ),
            _content_with_claim(
                ["LSTM"], "LSTM shows no improvement in forecasting accuracy over ARIMA"
            ),
        ]
    )
    svc = _service(FakeParser(docs), llm, graph)
    a = await svc.ingest_pdf("a.pdf", _pdf("A"))
    assert a.status == JobStatus.SUCCEEDED
    assert await svc.get_claim_relations("CONTRADICTS") == []  # nothing to clash with yet

    b = await svc.ingest_pdf("b.pdf", _pdf("B"))
    assert b.status == JobStatus.SUCCEEDED
    contradictions = await svc.get_claim_relations("CONTRADICTS")
    assert len(contradictions) == 1
    edge = contradictions[0]
    assert {edge.source_paper, edge.target_paper} == {a.paper_id, b.paper_id}
    # Persisted to the graph as a first-class edge, not just mirrored.
    assert any("CONTRADICTS" in q for q, _ in graph.calls)
    # Re-running the incremental pass is idempotent in the mirror.
    card_b = await svc.cards.get(b.paper_id)
    assert card_b is not None
    await svc._detect_relations_incremental(card_b)
    assert len(await svc.get_claim_relations("CONTRADICTS")) == 1


async def test_incremental_contradictions_can_be_disabled() -> None:
    docs = {
        "a.pdf": _doc("LSTM improves copper forecasting", "LSTM", ["10.1/a"]),
        "b.pdf": _doc("A critical replication of neural commodity models", "LSTM", ["10.1/b"]),
    }
    llm = ScriptedLLM(
        [
            _content_with_claim(["LSTM"], "LSTM significantly improves forecasting accuracy"),
            _content_with_claim(["LSTM"], "LSTM shows no improvement in forecasting accuracy"),
        ]
    )
    svc = _service(FakeParser(docs), llm, FakeGraphStore())
    svc.settings = svc.settings.model_copy(update={"incremental_contradictions": False})
    await svc.ingest_pdf("a.pdf", _pdf("A"))
    await svc.ingest_pdf("b.pdf", _pdf("B"))
    assert await svc.get_claim_relations("CONTRADICTS") == []


# ----------------------------------------------------------- calibrator sampling
async def test_calibrator_refit_is_bounded_on_large_corpora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All-pairs calibration is O(n^2) on EVERY ingest; the refit must subsample.
    import numpy as np
    from lattice.graph.similarity import PaperFeatures
    from lattice.ingestion import service as service_mod

    svc = _service(FakeParser({}), ScriptedLLM([]), FakeGraphStore())
    rng_vecs = [np.array([1.0, float(i % 7), float(i % 3)], dtype=float) for i in range(500)]
    for i, v in enumerate(rng_vecs):
        svc._features[f"p{i}"] = PaperFeatures(paper_id=f"p{i}", specter=v)

    calls = {"n": 0}
    real_cosine = service_mod.cosine

    def counting_cosine(a, b):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_cosine(a, b)

    monkeypatch.setattr(service_mod, "cosine", counting_cosine)
    calibrator = svc._fit_calibrator()
    assert calibrator.fitted
    cap = svc._CALIBRATION_VECS
    assert calls["n"] <= cap * (cap - 1) // 2, "refit must be bounded, not all-pairs"


# --------------------------------------------------- citation ids + references
class StubEnricher:
    def __init__(self, extra: dict[str, object]) -> None:
        self._extra = extra

    async def enrich(self, card):  # type: ignore[no-untyped-def]
        return dict(self._extra)


async def test_openalex_reference_space_direct_citation() -> None:
    # Paper A is known by its OpenAlex URL; paper B's references (as OpenAlex
    # returns them: full URLs) include that URL. Direct-citation detection must
    # match and write a CITES edge - this was structurally impossible before the
    # paper's own openalex_id was captured.
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    graph = FakeGraphStore()
    docs = {
        "a.pdf": _doc("LSTM copper forecasting", "LSTM attention", []),
        "b.pdf": _doc("LSTM copper prediction citing A", "LSTM gru", []),
    }
    svc = _service(
        FakeParser(docs),
        ScriptedLLM([_content(["LSTM"]), _content(["LSTM", "GRU"])]),
        graph,
        vectors=shared_vectors,
        cards=shared_cards,
    )
    svc.enricher = StubEnricher({"openalex_id": "https://openalex.org/W111"})
    a = await svc.ingest_pdf("a.pdf", _pdf("A"))
    # A's card carries its OpenAlex id and exposes it in the citation-id space.
    card_a = await shared_cards.get(a.paper_id)
    assert card_a is not None and card_a.openalex_id == "https://openalex.org/W111"
    assert "https://openalex.org/W111" in card_a.external_ids

    svc.enricher = StubEnricher({"reference_ids": ["https://openalex.org/W111"]})
    b = await svc.ingest_pdf("b.pdf", _pdf("B"))
    assert b.status == JobStatus.SUCCEEDED
    assert any("CITES" in q for q, _ in graph.calls), "direct citation must be detected"
    assert (b.paper_id, a.paper_id) in svc._cites


async def test_references_and_external_ids_survive_restart() -> None:
    # The FULL citation state now rehydrates: reference sets (bibliographic
    # coupling) and external ids (direct-citation detection) both persist.
    shared_cards = InMemoryCardStore()
    shared_vectors = InMemoryVectorStore()
    docs1 = {"a.pdf": _doc("LSTM copper forecasting", "LSTM attention", [])}
    svc1 = _service(
        FakeParser(docs1),
        ScriptedLLM([_content(["LSTM", "attention"])]),
        FakeGraphStore(),
        vectors=shared_vectors,
        cards=shared_cards,
    )
    svc1.enricher = StubEnricher(
        {"reference_ids": ["DOI:10.1/lstm-orig"], "openalex_id": "https://openalex.org/W42"}
    )
    a = await svc1.ingest_pdf("a.pdf", _pdf("A"))

    # Restart: fresh service, same stores; B's references cite A's OpenAlex URL.
    graph2 = FakeGraphStore()
    docs2 = {"b.pdf": _doc("LSTM copper prediction citing A", "LSTM gru", [])}
    svc2 = _service(
        FakeParser(docs2),
        ScriptedLLM([_content(["LSTM", "GRU"])]),
        graph2,
        vectors=shared_vectors,
        cards=shared_cards,
    )
    svc2.enricher = StubEnricher({"reference_ids": ["https://openalex.org/W42"]})
    b = await svc2.ingest_pdf("b.pdf", _pdf("B"))
    assert b.status == JobStatus.SUCCEEDED
    # A's reference set and external ids were rehydrated from the store.
    assert svc2._features[a.paper_id].references == {"DOI:10.1/lstm-orig"}
    assert "https://openalex.org/W42" in svc2._external_ids[a.paper_id]
    # And the cross-restart direct citation fired.
    assert any("CITES" in q for q, _ in graph2.calls)


async def test_entity_resolution_receives_embeddings() -> None:
    # The ambiguous-band embedding tiebreak needs vectors at resolve time; the
    # wiring must pass one for every method and dataset name.
    docs = {"a.pdf": _doc("LSTM copper forecasting", "LSTM attention", [])}
    svc = _service(FakeParser(docs), ScriptedLLM([_content(["LSTM"])]), FakeGraphStore())
    seen: list[object] = []
    assert svc.method_resolver is not None
    real_resolve = svc.method_resolver.resolve

    def spy(name: str, embedding=None):  # type: ignore[no-untyped-def]
        seen.append(embedding)
        return real_resolve(name, embedding=embedding)

    svc.method_resolver.resolve = spy  # type: ignore[method-assign]
    await svc.ingest_pdf("a.pdf", _pdf("A"))
    assert seen, "methods were resolved"
    assert all(e is not None for e in seen), "every resolution must carry an embedding"
