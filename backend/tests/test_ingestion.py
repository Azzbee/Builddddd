from __future__ import annotations

import pytest
from lattice.core.errors import (
    CorruptedPdfError,
    CostCapExceeded,
    DuplicateError,
    PaywalledStubError,
    ScannedPdfError,
)
from lattice.ingestion.chunker import chunk_document
from lattice.ingestion.dedup import CorpusIndex, PaperIdentity
from lattice.ingestion.models import (
    IngestJob,
    JobStage,
    JobStatus,
    ParsedDocument,
    ParsedSection,
    SourceType,
)
from lattice.ingestion.pdf_utils import classify_pdf, looks_like_paywall_stub
from lattice.ingestion.pipeline import IngestionPipeline, PipelineContext


# --------------------------------------------------------------------------- dedup
def _idx() -> CorpusIndex:
    return CorpusIndex.from_identities(
        [
            PaperIdentity(
                paper_id="p1",
                title="Deep Learning for Copper Price Forecasting",
                authors=["Jane Doe", "Carlos Ng"],
                doi="10.1234/abcd.2024",
                arxiv_id="2401.00001",
                content_hash="hash-1",
            )
        ]
    )


def test_dedup_by_content_hash() -> None:
    r = _idx().find_duplicate(PaperIdentity("x", "Totally different", content_hash="hash-1"))
    assert r.is_duplicate and r.reason == "content_hash" and r.existing_paper_id == "p1"


def test_dedup_by_doi_and_arxiv() -> None:
    idx = _idx()
    assert (
        idx.find_duplicate(PaperIdentity("x", "T", doi="https://doi.org/10.1234/ABCD.2024")).reason
        == "doi"
    )
    assert (
        idx.find_duplicate(PaperIdentity("x", "T", arxiv_id="arXiv:2401.00001v3")).reason == "arxiv"
    )


def test_dedup_title_author_fuzzy() -> None:
    # Preprint vs published: slightly reformatted title, same authors.
    cand = PaperIdentity(
        "x", "Deep learning for copper-price forecasting", authors=["J. Doe", "C. Ng"]
    )
    r = _idx().find_duplicate(cand)
    assert r.is_duplicate and r.reason == "title_author_fuzzy"


def test_dedup_no_false_positive_on_different_paper() -> None:
    cand = PaperIdentity("x", "Reinforcement learning for robotics", authors=["A. Other"])
    assert not _idx().find_duplicate(cand).is_duplicate


# --------------------------------------------------------------------------- chunker
def _doc() -> ParsedDocument:
    long_body = " ".join(f"Sentence number {i} about forecasting copper prices." for i in range(80))
    return ParsedDocument(
        title="t",
        abstract="This is the abstract about LSTM forecasting.",
        sections=[
            ParsedSection(section_id="s1", title="Introduction", text=long_body),
            ParsedSection(
                section_id="s2",
                title="Methodology",
                text="We use an LSTM. It is trained on daily data.",
            ),
        ],
    )


def test_chunker_respects_sections_and_anchors() -> None:
    chunks = chunk_document(_doc(), "p1", max_tokens=64, overlap_tokens=8)
    assert chunks, "should produce chunks"
    assert all(c.paper_id == "p1" for c in chunks)
    # Abstract is its own chunk.
    assert any(c.section_id == "abstract" for c in chunks)
    # Chunks never mix sections.
    for c in chunks:
        assert c.section_id in {"abstract", "s1", "s2"}
    # The long introduction must split into multiple chunks at this token budget.
    intro_chunks = [c for c in chunks if c.section_id == "s1"]
    assert len(intro_chunks) >= 2
    # Ordinals are unique and monotonic.
    ordinals = [c.ordinal for c in chunks]
    assert ordinals == sorted(ordinals)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunker_hard_splits_oversized_sentence_after_a_short_one() -> None:
    # Regression: an oversized sentence that arrives while a window is already open
    # used to be appended whole and emitted over the char budget. It must be
    # hard-split regardless of prior window state.
    max_tokens = 64
    max_chars = max_tokens * 4  # _CHARS_PER_TOKEN
    giant = "x" * (max_chars * 3)  # one sentence, no sentence terminators
    doc = ParsedDocument(
        title="t",
        sections=[
            ParsedSection(section_id="s1", title="Body", text=f"A short lead sentence. {giant}."),
        ],
    )
    chunks = chunk_document(doc, "p1", max_tokens=max_tokens, overlap_tokens=8, min_chars=1)
    body = [c for c in chunks if c.section_id == "s1"]
    assert body, "should produce chunks for the section"
    assert all(len(c.text) <= max_chars for c in body), "no chunk may exceed the char budget"
    # The giant sentence must have been split into several windows.
    assert len(body) >= 3


def test_chunker_deterministic_ids() -> None:
    a = chunk_document(_doc(), "p1")
    b = chunk_document(_doc(), "p1")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunker_carries_page_for_deep_linking() -> None:
    doc = ParsedDocument(
        title="t",
        abstract="Abstract text about forecasting that is sufficiently long.",
        sections=[
            ParsedSection(
                section_id="s1",
                title="Results",
                text="We beat the ARIMA baseline on RMSE across all forecast horizons tested.",
                page=7,
            ),
        ],
    )
    chunks = chunk_document(doc, "p1")
    abstract = next(c for c in chunks if c.section_id == "abstract")
    results = next(c for c in chunks if c.section_id == "s1")
    assert abstract.page == 1  # abstract anchors to page 1
    assert results.page == 7  # section page propagates to its chunks


# --------------------------------------------------------------------------- pdf utils
def _valid_pdf_bytes() -> bytes:
    return b"%PDF-1.7\n" + b"x" * 5000


def test_classify_pdf_corrupted() -> None:
    with pytest.raises(CorruptedPdfError):
        classify_pdf(b"not a pdf at all", extract_text=lambda d: "")
    with pytest.raises(CorruptedPdfError):
        classify_pdf(b"", extract_text=lambda d: "")


def test_classify_pdf_scanned() -> None:
    with pytest.raises(ScannedPdfError):
        classify_pdf(_valid_pdf_bytes(), extract_text=lambda d: "tiny")


def test_classify_pdf_paywalled() -> None:
    stub = "Please sign in to read this article. Institutional login required."
    with pytest.raises(PaywalledStubError):
        classify_pdf(_valid_pdf_bytes(), extract_text=lambda d: stub)


def test_classify_pdf_ok_returns_text() -> None:
    text = "Real paper content. " * 60
    assert classify_pdf(_valid_pdf_bytes(), extract_text=lambda d: text) == text


def test_looks_like_paywall_stub() -> None:
    assert looks_like_paywall_stub("purchase pdf", 1000)
    assert not looks_like_paywall_stub("normal content " * 100, 200_000)


def test_classify_pdf_default_extractor_real_pdf() -> None:
    # Exercise the real pypdf-backed default extractor (no injected extract_text):
    # a blank-page PDF parses fine but yields no text -> ScannedPdfError, not
    # CorruptedPdfError. Guards against a missing/mislabeled parsing dependency.
    import io

    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    data = buf.getvalue() + b" " * max(0, 3000 - buf.getbuffer().nbytes)
    with pytest.raises(ScannedPdfError):
        classify_pdf(data)


def test_missing_pypdf_is_config_error_not_corrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    # If pypdf is absent, the error must say "install the parsing extra", not
    # mislabel every upload as a corrupted file.
    import sys

    from lattice.core.errors import ParserUnavailableError

    monkeypatch.setitem(sys.modules, "pypdf", None)  # makes `from pypdf import ...` fail
    with pytest.raises(ParserUnavailableError, match="parsing"):
        classify_pdf(_valid_pdf_bytes())


# --------------------------------------------------------------------------- pipeline
def _job() -> IngestJob:
    return IngestJob(job_id="j1", source_type=SourceType.FILE, source_ref="paper.pdf")


async def test_pipeline_happy_path_runs_all_stages() -> None:
    visited: list[JobStage] = []

    def make(stage: JobStage):
        async def handler(ctx: PipelineContext) -> None:
            visited.append(stage)

        return handler

    handlers = {
        s: make(s)
        for s in (
            JobStage.PARSING,
            JobStage.EXTRACTING,
            JobStage.ENRICHING,
            JobStage.EMBEDDING,
            JobStage.LINKING,
        )
    }
    pipe = IngestionPipeline(handlers)
    job = await pipe.run(PipelineContext(job=_job()))
    assert job.status == JobStatus.SUCCEEDED
    assert job.stage == JobStage.DONE
    assert job.progress == 1.0
    assert visited == [
        JobStage.PARSING,
        JobStage.EXTRACTING,
        JobStage.ENRICHING,
        JobStage.EMBEDDING,
        JobStage.LINKING,
    ]


async def test_pipeline_duplicate_short_circuits() -> None:
    async def parse(ctx: PipelineContext) -> None:
        ctx.extra["duplicate_of"] = "p1"
        raise DuplicateError("already have it")

    pipe = IngestionPipeline({JobStage.PARSING: parse})
    job = await pipe.run(PipelineContext(job=_job()))
    assert job.status == JobStatus.DUPLICATE
    assert job.paper_id == "p1"
    assert job.stage == JobStage.RECEIVED  # never advanced


async def test_pipeline_cost_cap_pauses_resumably() -> None:
    calls = {"n": 0}

    async def parse(ctx: PipelineContext) -> None:
        return None

    async def extract(ctx: PipelineContext) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise CostCapExceeded("daily cap")

    handlers = {JobStage.PARSING: parse, JobStage.EXTRACTING: extract}
    pipe = IngestionPipeline(handlers)
    ctx = PipelineContext(job=_job())
    job = await pipe.run(ctx)
    assert job.status == JobStatus.PAUSED
    assert job.stage == JobStage.PARSING  # completed parse, paused before extract
    assert job.error_code == "cost_cap_exceeded"

    # Resume: same context, second attempt at extract succeeds, then must finish.
    full = {
        JobStage.PARSING: parse,
        JobStage.EXTRACTING: extract,
        JobStage.ENRICHING: parse,
        JobStage.EMBEDDING: parse,
        JobStage.LINKING: parse,
    }
    job2 = await IngestionPipeline(full).run(ctx)
    assert job2.status == JobStatus.SUCCEEDED
    assert job2.stage == JobStage.DONE


async def test_pipeline_failure_records_code() -> None:
    async def parse(ctx: PipelineContext) -> None:
        raise CorruptedPdfError("bad pdf")

    pipe = IngestionPipeline({JobStage.PARSING: parse})
    job = await pipe.run(PipelineContext(job=_job()))
    assert job.status == JobStatus.FAILED
    assert job.error_code == "corrupted_pdf"
    assert job.attempts == 1
