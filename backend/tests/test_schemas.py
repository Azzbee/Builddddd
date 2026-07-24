from __future__ import annotations

import pytest
from lattice.extraction.schemas import (
    Author,
    DatasetRef,
    LLMPaperCardContent,
    Methodology,
    PaperCard,
    PaperType,
    ReproSignals,
    Result,
)
from pydantic import ValidationError


def _methodology() -> Methodology:
    return Methodology(
        approach_summary="LSTM with attention for price forecasting",
        method_family=["deep learning", "Deep Learning "],  # dedupe + strip
        techniques=["LSTM", "attention", "lstm"],  # dedupe normalized
        baselines=["ARIMA"],
        reproducibility=ReproSignals(code_available=True),
    )


def test_methodology_dedupes_normalized_lists() -> None:
    m = _methodology()
    assert m.method_family == ["deep learning"]
    assert m.techniques == ["LSTM", "attention"]


def test_papercard_normalized_helpers() -> None:
    card = PaperCard(
        paper_id="p1",
        title="A paper",
        methodology=_methodology(),
        methods_taxonomy=["LSTM", "VAR"],
        datasets=[DatasetRef(name="LME Copper"), DatasetRef(name="lme copper")],
    )
    assert card.normalized_methods == {"lstm", "var", "attention"}
    assert card.normalized_datasets == {"lme copper"}


def test_papercard_year_and_confidence_validation() -> None:
    card = PaperCard(
        paper_id="p1",
        title="t",
        methodology=_methodology(),
        year=3500,  # implausible -> None
        confidence=2.0,  # clamped
    )
    assert card.year is None
    assert card.confidence == 1.0


def test_author_normalized_name() -> None:
    assert Author(name="  Émile  Zola ").normalized_name == "emile zola"


def test_llm_boundary_tolerant_but_papercard_strict() -> None:
    # LLM-facing models ignore unknown keys (weak models add junk); the internal
    # PaperCard stays extra="forbid" so OUR bugs are still caught loudly.
    r = Result.model_validate({"claim": "x", "bogus": "y"})
    assert r.claim == "x" and not hasattr(r, "bogus")
    with pytest.raises(ValidationError):
        PaperCard(paper_id="p", title="t", methodology=_methodology(), bogus=1)  # type: ignore[call-arg]


def test_llm_boundary_null_means_absent() -> None:
    # Observed live from a local 7B model: null for "empty" list/scalar fields.
    content = LLMPaperCardContent.model_validate(
        {
            "problem_statement": "P.",
            "methodology": {
                "approach_summary": "A.",
                "baselines": None,
                "evaluation_protocol": None,
            },
            "limitations": None,
            "self_confidence": None,
            "key_results": [{"claim": "c", "evidence_location": None}],
        }
    )
    assert content.methodology.baselines == []
    assert content.methodology.evaluation_protocol is None  # genuinely optional stays None
    assert content.limitations == []
    assert content.self_confidence == 0.5  # default applied
    assert content.key_results[0].evidence_location == ""


def test_llm_boundary_null_required_still_fails() -> None:
    with pytest.raises(ValidationError):
        LLMPaperCardContent.model_validate(
            {"problem_statement": None, "methodology": {"approach_summary": "A."}}
        )


def test_llm_boundary_wraps_bare_scalar_into_list() -> None:
    content = LLMPaperCardContent.model_validate(
        {
            "problem_statement": "P.",
            "methodology": {"approach_summary": "A.", "techniques": "LSTM"},
            "contributions": "a single contribution",
        }
    )
    assert content.contributions == ["a single contribution"]
    assert content.methodology.techniques == ["LSTM"]


def test_paper_type_case_insensitive() -> None:
    assert PaperType(" EMPIRICAL ") is PaperType.EMPIRICAL
    assert PaperType("Survey") is PaperType.SURVEY
    with pytest.raises(ValueError):
        PaperType("poem")  # a genuinely wrong value still fails -> repair loop


def test_llm_boundary_drops_degenerate_list_entries() -> None:
    # Verbatim live failure (local 7B, Dreamer paper): one all-null dataset entry
    # burned every repair attempt and killed the whole extraction. Degenerate
    # entries are noise ("nothing here"); the card must survive without them.
    content = LLMPaperCardContent.model_validate(
        {
            "problem_statement": "P.",
            "methodology": {"approach_summary": "A."},
            "datasets": [
                {
                    "name": None,
                    "source": None,
                    "size": None,
                    "is_public": None,
                    "url": None,
                    "evidence_location": None,
                },
                {"name": "DeepMind Control Suite", "source": None},
                None,
            ],
            "key_results": [{"claim": None, "metric": "RMSE"}, {"claim": "real claim"}],
            "limitations": ["a", None, "b"],
        }
    )
    assert [d.name for d in content.datasets] == ["DeepMind Control Suite"]
    assert [r.claim for r in content.key_results] == ["real claim"]
    assert content.limitations == ["a", "b"]


def test_llm_boundary_coerces_bare_string_list_entries() -> None:
    # "datasets": ["LME Copper"] -> the element's single required field.
    content = LLMPaperCardContent.model_validate(
        {
            "problem_statement": "P.",
            "methodology": {"approach_summary": "A."},
            "datasets": ["LME Copper"],
            "key_results": ["beats ARIMA"],
        }
    )
    assert content.datasets[0].name == "LME Copper"
    assert content.key_results[0].claim == "beats ARIMA"


def test_llm_content_to_card_merges_identity_and_meta() -> None:
    content = LLMPaperCardContent(
        problem_statement="Forecasting is hard.",
        methodology=_methodology(),
        datasets=[DatasetRef(name="LME")],
        key_results=[Result(claim="beats ARIMA", metric="RMSE", evidence_location="Table 2")],
        paper_type=PaperType.EMPIRICAL,
        methods_taxonomy=["LSTM"],
        self_confidence=0.8,
    )
    card = content.to_card(
        identity={
            "paper_id": "p1",
            "title": "Forecasting copper",
            "authors": [Author(name="A. Author")],
            "year": 2024,
            "doi": "10.1/x",
        },
        meta={
            "extraction_model": "claude-haiku-4-5-20251001",
            "extraction_version": "papercard_v1",
            "confidence": 0.8,
            "needs_review": False,
        },
    )
    assert card.title == "Forecasting copper"
    assert card.paper_type is PaperType.EMPIRICAL
    assert card.key_results[0].evidence_location == "Table 2"
    assert card.confidence == 0.8


def test_external_ids_cover_every_citation_space() -> None:
    card = PaperCard(
        paper_id="p1",
        title="t",
        methodology=_methodology(),
        doi="10.1/x",
        arxiv_id="2101.00001",
        s2_paper_id="s2abc",
        openalex_id="https://openalex.org/W99",
    )
    assert card.external_ids == {"DOI:10.1/x", "2101.00001", "s2abc", "https://openalex.org/W99"}
    assert PaperCard(paper_id="p2", title="t", methodology=_methodology()).external_ids == set()
