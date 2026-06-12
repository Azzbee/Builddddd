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


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Result(claim="x", bogus="y")  # type: ignore[call-arg]


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
