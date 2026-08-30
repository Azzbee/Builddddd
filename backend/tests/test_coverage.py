from __future__ import annotations

from lattice.config import Settings
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.db.cards import InMemoryCardStore
from lattice.db.vector import InMemoryVectorStore, VectorRecord
from lattice.embeddings.chunks import ChunkEmbedder
from lattice.embeddings.specter2 import Specter2Embedder
from lattice.extraction.schemas import DatasetRef, Methodology, PaperCard
from lattice.ingestion.models import ParsedDocument
from lattice.ingestion.service import IngestionService
from lattice.landscape.coverage import (
    COVERED_MIN,
    DEFAULT_PROBE_LIMIT,
    PARTIAL_MIN,
    CoverageState,
    Probe,
    ProbeEvidence,
    ProbeSource,
    QuestionMention,
    blind_spots,
    content_tokens,
    facet_question,
    generate_probes,
    score_probe,
    summarize_coverage,
)
from lattice.landscape.matrix import CellState, MatrixCell
from lattice.landscape.quadrants import OpenProblemCluster


# --------------------------------------------------------------------------- tokens
def test_content_tokens_drops_interrogatives_and_depluralizes() -> None:
    toks = content_tokens("How does LSTM perform on the copper datasets?")
    assert "lstm" in toks
    assert "copper" in toks
    assert "dataset" in toks  # de-pluralized
    for stop in ("how", "does", "the", "on", "perform"):
        assert stop not in toks


def test_content_tokens_empty_text() -> None:
    assert content_tokens("") == set()
    assert content_tokens("how is it?") == set()


# --------------------------------------------------------------------------- templates
def test_facet_question_uses_pair_template_in_both_directions() -> None:
    assert facet_question("method", "dataset", "lstm", "lme copper") == (
        "How does lstm perform on lme copper?"
    )
    # The reversed pair reuses the same template with the values swapped.
    assert facet_question("dataset", "method", "lme copper", "lstm") == (
        "How does lstm perform on lme copper?"
    )


def test_facet_question_falls_back_for_unknown_pairs() -> None:
    assert facet_question("venue", "author", "a", "b") == "What is known about a and b?"


# --------------------------------------------------------------------------- generation
def _gap_cell(row: str, col: str, gap_score: float) -> MatrixCell:
    return MatrixCell(
        row=row,
        col=col,
        paper_ids=[],
        latest_year=None,
        sparkline={},
        state=CellState.EMPTY,
        gap_score=gap_score,
    )


def test_generate_probes_covers_all_three_sources() -> None:
    bank = generate_probes(
        [QuestionMention("p1", "Can attention models forecast copper prices?")],
        [
            OpenProblemCluster(
                canonical_text="incorporate macroeconomic covariates",
                paper_ids=["p1", "p2"],
                frequency=2,
                latest_year=2022,
                score=2.0,
            )
        ],
        [_gap_cell("diffusion", "lme aluminium", 0.4)],
    )
    sources = {p.source for p in bank.probes}
    assert sources == {
        ProbeSource.RESEARCH_QUESTION,
        ProbeSource.OPEN_PROBLEM,
        ProbeSource.FACET_CROSS,
    }
    assert not bank.truncated
    assert bank.dropped == {}


def test_generate_probes_dedupes_and_counts_salience() -> None:
    mentions = [
        QuestionMention("p1", "Do transformers beat LSTM on copper?"),
        QuestionMention("p2", "Do transformers beat LSTM on copper?"),
        QuestionMention("p2", "Do transformers beat LSTM on copper?"),  # same paper, once
    ]
    bank = generate_probes(mentions)
    assert len(bank.probes) == 1
    probe = bank.probes[0]
    assert probe.origin_paper_ids == ("p1", "p2")
    assert probe.salience == 2 / 3


def test_generate_probes_skips_contentless_questions() -> None:
    bank = generate_probes([QuestionMention("p1", "How does it work?")])
    assert bank.probes == []


def test_generate_probes_interleaves_sources_under_the_cap() -> None:
    questions = [
        QuestionMention("p1", f"Does method {i} beat the copper baseline?") for i in range(10)
    ]
    gaps = [_gap_cell(f"method{i}", "lme copper", 0.5) for i in range(10)]
    bank = generate_probes(questions, [], gaps, limit=4)
    assert len(bank.probes) == 4
    # Round-robin: neither source is crowded out by the other.
    by_source = dict.fromkeys(ProbeSource, 0)
    for p in bank.probes:
        by_source[p.source] += 1
    assert by_source[ProbeSource.RESEARCH_QUESTION] == 2
    assert by_source[ProbeSource.FACET_CROSS] == 2
    assert bank.truncated
    assert bank.dropped == {
        str(ProbeSource.RESEARCH_QUESTION): 8,
        str(ProbeSource.FACET_CROSS): 8,
    }
    assert bank.generated[str(ProbeSource.FACET_CROSS)] == 10


def test_generate_probes_empty_corpus() -> None:
    bank = generate_probes()
    assert bank.probes == []
    assert bank.generated == {}
    assert bank.dropped == {}
    assert not bank.truncated


def test_generate_probes_zero_limit_reports_everything_dropped() -> None:
    bank = generate_probes([QuestionMention("p1", "Does LSTM beat ARIMA on copper?")], limit=0)
    assert bank.probes == []
    assert bank.dropped == {str(ProbeSource.RESEARCH_QUESTION): 1}


def test_default_probe_limit_is_bounded() -> None:
    assert 0 < DEFAULT_PROBE_LIMIT <= 200


# --------------------------------------------------------------------------- scoring
def _probe(text: str = "How does LSTM perform on LME copper?", **kw: object) -> Probe:
    kwargs: dict[str, object] = {"source": ProbeSource.FACET_CROSS}
    kwargs.update(kw)
    return Probe(text=text, **kwargs)  # type: ignore[arg-type]


def _evidence(paper: str, text: str, score: float) -> ProbeEvidence:
    return ProbeEvidence(
        paper_id=paper,
        title="A paper",
        text=text,
        score=score,
        section="Results",
        evidence_location="Table 3",
        page=8,
    )


def test_score_probe_with_no_evidence_is_uncovered() -> None:
    result = score_probe(_probe(), [])
    assert result.state is CoverageState.UNCOVERED
    assert result.coverage == 0.0
    assert result.retrieval == result.support == result.grounding == 0.0
    assert result.best_evidence is None
    assert set(result.missing_terms) == {"lstm", "lme", "copper"}
    assert result.pressure > 0


def test_score_probe_ignores_zero_score_evidence() -> None:
    result = score_probe(_probe(), [_evidence("p1", "unrelated text", 0.0)])
    assert result.state is CoverageState.UNCOVERED
    assert result.best_evidence is None


def test_score_probe_well_answered_question_is_covered() -> None:
    evidence = [
        _evidence("p1", "LSTM forecasts of LME copper beat ARIMA", 0.92),
        _evidence("p2", "LSTM applied to LME copper prices", 0.88),
        _evidence("p3", "LME copper forecasting with LSTM models", 0.85),
    ]
    result = score_probe(_probe(), evidence)
    assert result.grounding == 1.0
    assert result.support == 1.0
    assert result.coverage >= COVERED_MIN
    assert result.state is CoverageState.COVERED
    assert result.supporting_papers == ["p1", "p2", "p3"]
    assert result.missing_terms == []
    assert result.best_evidence is not None
    assert result.best_evidence.paper_id == "p1"


def test_score_probe_reports_the_terms_the_corpus_cannot_ground() -> None:
    evidence = [_evidence("p1", "LSTM forecasting results for gold", 0.9)]
    result = score_probe(_probe(), evidence)
    assert result.missing_terms == ["copper", "lme"]
    assert 0.0 < result.grounding < 1.0
    assert result.state is not CoverageState.COVERED


def test_score_probe_support_uses_distinct_papers_near_the_top() -> None:
    evidence = [
        _evidence("p1", "LSTM on LME copper", 0.9),
        _evidence("p1", "LSTM on LME copper again", 0.89),  # same paper, no extra breadth
        _evidence("p2", "LSTM on LME copper elsewhere", 0.2),  # too far below the top hit
    ]
    result = score_probe(_probe(), evidence)
    assert result.supporting_papers == ["p1"]
    assert result.support == 1 / 3


def test_score_probe_retrieval_component_is_clipped() -> None:
    result = score_probe(_probe(), [_evidence("p1", "LSTM LME copper", 3.0)])
    assert result.retrieval == 1.0


def test_pressure_ranks_blind_spot_probes_above_known_open_problems() -> None:
    facet = score_probe(_probe(source=ProbeSource.FACET_CROSS, salience=0.5), [])
    question = score_probe(_probe(source=ProbeSource.RESEARCH_QUESTION, salience=0.5), [])
    assert facet.pressure > question.pressure


def test_pressure_rises_with_salience() -> None:
    loud = score_probe(_probe(salience=1.0), [])
    quiet = score_probe(_probe(salience=0.0), [])
    assert loud.pressure > quiet.pressure


# --------------------------------------------------------------------------- reporting
def test_blind_spots_excludes_covered_and_ranks_by_pressure() -> None:
    covered = score_probe(
        _probe("How does LSTM perform on LME copper?"),
        [
            _evidence("p1", "LSTM on LME copper", 0.95),
            _evidence("p2", "LSTM on LME copper", 0.94),
            _evidence("p3", "LSTM on LME copper", 0.93),
        ],
    )
    weak = score_probe(_probe("How does diffusion perform on LME nickel?", salience=0.2), [])
    strong = score_probe(_probe("How does diffusion perform on LME tin?", salience=1.0), [])
    ranked = blind_spots([covered, weak, strong])
    assert [r.probe.text for r in ranked] == [strong.probe.text, weak.probe.text]


def test_blind_spots_respects_the_limit() -> None:
    results = [score_probe(_probe(f"How does m{i} perform on copper?"), []) for i in range(5)]
    assert len(blind_spots(results, limit=2)) == 2


def test_summarize_coverage_reports_index_and_histograms() -> None:
    covered = score_probe(
        _probe("How does LSTM perform on LME copper?"),
        [
            _evidence("p1", "LSTM on LME copper", 0.95),
            _evidence("p2", "LSTM on LME copper", 0.94),
            _evidence("p3", "LSTM on LME copper", 0.93),
        ],
    )
    uncovered = score_probe(_probe("How does diffusion perform on LME nickel?"), [])
    summary = summarize_coverage([covered, uncovered])
    assert summary["probe_count"] == 2
    assert summary["blind_spot_ratio"] == 0.5
    by_state = summary["by_state"]
    assert isinstance(by_state, dict)
    assert by_state[str(CoverageState.COVERED)] == 1
    assert by_state[str(CoverageState.UNCOVERED)] == 1
    assert 0.0 < float(str(summary["coverage_index"])) < 1.0


def test_summarize_coverage_empty() -> None:
    summary = summarize_coverage([])
    assert summary["probe_count"] == 0
    assert summary["coverage_index"] == 0.0
    assert summary["blind_spot_ratio"] == 0.0


def test_probe_result_json_is_self_describing() -> None:
    result = score_probe(
        _probe(salience=0.5, facet_cell=("diffusion", "lme nickel")),
        [_evidence("p1", "LSTM on LME copper", 0.9)],
    )
    js = result.to_json()
    assert js["source"] == str(ProbeSource.FACET_CROSS)
    assert js["facet_cell"] == ["diffusion", "lme nickel"]
    components = js["components"]
    assert isinstance(components, dict)
    assert set(components) == {"retrieval", "support", "grounding"}
    best = js["best_evidence"]
    assert isinstance(best, dict)
    assert best["paper_id"] == "p1"
    assert best["page"] == 8


# --------------------------------------------------------------------------- service
class ScriptedLLM:
    """The coverage path is retrieval-only, so the model must never be called."""

    async def complete(
        self, model: str, messages: list[LLMMessage], **kwargs: object
    ) -> LLMResponse:
        raise AssertionError("question_coverage must not call the LLM")


class NullParser:
    async def process_fulltext(
        self, pdf_bytes: bytes, filename: str = "paper.pdf"
    ) -> ParsedDocument:
        raise AssertionError("question_coverage must not parse PDFs")


def _card(
    paper_id: str,
    *,
    methods: list[str],
    datasets: list[str],
    questions: list[str],
    future_work: list[str],
    year: int = 2023,
) -> PaperCard:
    return PaperCard(
        paper_id=paper_id,
        title=f"{methods[0]} forecasting of {datasets[0]}",
        year=year,
        problem_statement=f"Forecasting {datasets[0]} with {methods[0]} is hard.",
        research_questions=questions,
        methodology=Methodology(approach_summary=f"{methods[0]} forecaster", techniques=methods),
        datasets=[DatasetRef(name=d) for d in datasets],
        future_work=future_work,
        domains=["commodity markets"],
        methods_taxonomy=methods,
    )


def _chunk(paper_id: str, title: str, text: str) -> VectorRecord:
    return VectorRecord(
        chunk_id=f"{paper_id}:{abs(hash(text)) % 10_000}",
        paper_id=paper_id,
        workspace_id="default",
        title=title,
        section_title="Results",
        text=text,
        embedding=[],
        evidence_location="Table 1",
        page=3,
    )


async def _service_with_corpus() -> IngestionService:
    """A two-paper corpus whose text answers one question and not another."""
    cards = InMemoryCardStore()
    vectors = InMemoryVectorStore()
    embedder = ChunkEmbedder(dim=128)
    svc = IngestionService(
        settings=Settings(),
        llm=ScriptedLLM(),
        parser=NullParser(),
        vectors=vectors,
        cards=cards,
        chunk_embedder=embedder,
        specter=Specter2Embedder(dim=64),
    )
    corpus = [
        _card(
            "p1",
            methods=["LSTM"],
            datasets=["LME Copper"],
            questions=["Can LSTM models forecast LME copper prices?"],
            future_work=["incorporate macroeconomic covariates"],
        ),
        _card(
            "p2",
            methods=["diffusion"],
            datasets=["COMEX Gold"],
            questions=["Do diffusion models calibrate COMEX gold scenarios?"],
            future_work=["reduce inference cost"],
        ),
    ]
    texts = {
        "p1": "LSTM models forecast LME copper prices and beat the ARIMA baseline.",
        "p2": "Diffusion models calibrate COMEX gold scenario forecasts under shocks.",
    }
    records = []
    for card in corpus:
        await cards.put_card(card)
        records.append(_chunk(card.paper_id, card.title, texts[card.paper_id]))
    for record, vec in zip(records, embedder.embed_texts([r.text for r in records]), strict=True):
        record.embedding = vec
    await vectors.upsert_chunks(records)
    return svc


async def test_question_coverage_reports_probes_and_blind_spots() -> None:
    svc = await _service_with_corpus()
    report = await svc.question_coverage()

    assert report["row_facet"] == "method"
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["probe_count"] > 0
    assert 0.0 <= float(str(summary["coverage_index"])) <= 1.0

    probes = report["probes"]
    assert isinstance(probes, list)
    sources = {p["source"] for p in probes}
    # Research questions, open problems, and the unasked facet crossings.
    assert sources == {
        str(ProbeSource.RESEARCH_QUESTION),
        str(ProbeSource.OPEN_PROBLEM),
        str(ProbeSource.FACET_CROSS),
    }

    # The crossing nobody studied (LSTM on gold, diffusion on copper) is a blind spot.
    spots = report["blind_spots"]
    assert isinstance(spots, list)
    crossings = {tuple(s["facet_cell"]) for s in spots if s["facet_cell"]}
    assert ("diffusion", "lme copper") in crossings or ("lstm", "comex gold") in crossings
    assert all(s["state"] != str(CoverageState.COVERED) for s in spots)


async def test_question_coverage_grounds_answerable_questions_better_than_gaps() -> None:
    svc = await _service_with_corpus()
    report = await svc.question_coverage()
    probes = report["probes"]
    assert isinstance(probes, list)
    by_text = {str(p["text"]): p for p in probes}

    answered = by_text["Can LSTM models forecast LME copper prices?"]
    components = answered["components"]
    assert isinstance(components, dict)
    assert components["grounding"] == 1.0  # every content term appears in the corpus

    unasked = next(
        p for p in probes if p["source"] == str(ProbeSource.FACET_CROSS) and p["missing_terms"]
    )
    assert unasked["coverage"] < answered["coverage"]


async def test_question_coverage_honours_the_probe_cap() -> None:
    svc = await _service_with_corpus()
    report = await svc.question_coverage(limit=2)
    probes = report["probes"]
    assert isinstance(probes, list)
    assert len(probes) == 2
    dropped = report["dropped_by_cap"]
    assert isinstance(dropped, dict)
    assert sum(dropped.values()) > 0  # the cap is reported, never silent


async def test_question_coverage_on_an_empty_corpus() -> None:
    svc = IngestionService(
        settings=Settings(),
        llm=ScriptedLLM(),
        parser=NullParser(),
        vectors=InMemoryVectorStore(),
        cards=InMemoryCardStore(),
        chunk_embedder=ChunkEmbedder(dim=64),
        specter=Specter2Embedder(dim=64),
    )
    report = await svc.question_coverage()
    assert report["probes"] == []
    assert report["blind_spots"] == []
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["probe_count"] == 0
    assert summary["coverage_index"] == 0.0


# --------------------------------------------------------------------------- edge cases
def test_generate_probes_skips_duplicate_and_contentless_open_problems() -> None:
    def cluster(text: str) -> OpenProblemCluster:
        return OpenProblemCluster(
            canonical_text=text,
            paper_ids=["p1"],
            frequency=1,
            latest_year=2023,
            score=1.0,
        )

    bank = generate_probes(
        open_problems=[
            cluster("incorporate macroeconomic covariates"),
            cluster("Incorporate macroeconomic covariates."),  # same after normalization
            cluster("do it"),  # no content tokens
        ]
    )
    assert [p.text for p in bank.probes] == ["incorporate macroeconomic covariates"]


def test_generate_probes_skips_gap_cells_that_render_the_same_question() -> None:
    bank = generate_probes(
        gap_cells=[
            _gap_cell("lstm", "lme copper", 0.5),
            _gap_cell("lstm", "LME Copper", 0.4),  # same question after normalization
        ]
    )
    # Deduped during generation, so it never reaches (or is blamed on) the cap.
    assert len(bank.probes) == 1
    assert bank.generated == {str(ProbeSource.FACET_CROSS): 1}
    assert bank.dropped == {}


def test_generate_probes_drops_a_question_already_supplied_by_another_source() -> None:
    shared = "incorporate macroeconomic covariates"
    bank = generate_probes(
        [QuestionMention("p1", shared)],
        [
            OpenProblemCluster(
                canonical_text=shared,
                paper_ids=["p2"],
                frequency=1,
                latest_year=2023,
                score=1.0,
            )
        ],
    )
    assert len(bank.probes) == 1
    assert bank.probes[0].source is ProbeSource.RESEARCH_QUESTION
    assert bank.dropped == {str(ProbeSource.OPEN_PROBLEM): 1}


def test_score_probe_partially_answered_question() -> None:
    # One paper, one of three terms grounded, middling retrieval: partial, not covered.
    result = score_probe(_probe(), [_evidence("p1", "LSTM training on equity indices", 0.5)])
    assert result.grounding == 1 / 3
    assert result.state is CoverageState.PARTIAL
    assert PARTIAL_MIN <= result.coverage < COVERED_MIN


def test_gap_salience_is_relative_to_the_strongest_gap() -> None:
    # gap_score is feasibility x adjacency x demand: a product that lands in the
    # low tenths on any real corpus. Used raw it buries facet crossings under
    # mention-count salience, so it is rescaled against the top gap in the set.
    bank = generate_probes(
        gap_cells=[
            _gap_cell("diffusion", "lme copper", 0.08),
            _gap_cell("garch", "lme copper", 0.02),
        ]
    )
    by_row = {p.facet_cell[0]: p.salience for p in bank.probes if p.facet_cell}
    assert by_row["diffusion"] == 1.0
    assert by_row["garch"] == 0.25


def test_gap_salience_is_zero_when_every_cell_scores_zero() -> None:
    bank = generate_probes(gap_cells=[_gap_cell("diffusion", "lme copper", 0.0)])
    assert bank.probes[0].salience == 0.0


def test_missing_terms_report_words_not_stems() -> None:
    # Matching de-pluralizes ("commodities" -> "commoditie"), but the stem is not
    # a word and missing_terms is read by a human.
    probe = _probe("How do diffusion models generalize across commodities?")
    result = score_probe(probe, [_evidence("p1", "Copper price forecasting", 0.6)])
    assert "commodities" in result.missing_terms
    assert "commoditie" not in result.missing_terms
    assert "across" not in result.missing_terms  # prose glue never becomes a term
