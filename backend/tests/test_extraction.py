from __future__ import annotations

import json

import pytest
from lattice.config import ExtractionSettings
from lattice.core.cost import CostTracker
from lattice.core.errors import SchemaValidationError
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.extraction.extractor import (
    completeness,
    extract_paper_card,
    render_prompt,
    score_confidence,
)
from lattice.extraction.prompts import available_versions, load_prompt, prompt_hash
from lattice.extraction.schemas import LLMPaperCardContent


class ScriptedLLM:
    """Returns queued responses in order, recording the models used."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.models: list[str] = []

    async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.models.append(model)
        text = self._responses.pop(0)
        return LLMResponse(text=text, input_tokens=1000, output_tokens=500, model=model)


def _content(**over) -> dict:
    base = {
        "problem_statement": "Forecasting copper prices is hard and underexplored.",
        "research_questions": ["Can LSTMs beat ARIMA?"],
        "methodology": {
            "approach_summary": "LSTM with attention trained on daily prices",
            "method_family": ["deep learning"],
            "techniques": ["LSTM", "attention"],
            "baselines": ["ARIMA"],
            "reproducibility": {"code_available": True},
        },
        "datasets": [{"name": "LME Copper", "source": "LME", "is_public": True}],
        "key_results": [
            {
                "claim": "beats ARIMA",
                "metric": "RMSE",
                "value": "0.12",
                "evidence_location": "Table 3",
            }
        ],
        "limitations": ["single commodity"],
        "contributions": ["new attention variant"],
        "future_work": ["test on other metals"],
        "paper_type": "empirical",
        "domains": ["commodity markets"],
        "methods_taxonomy": ["LSTM"],
        "self_confidence": 0.9,
    }
    base.update(over)
    return base


def _identity() -> dict:
    return {"paper_id": "p1", "title": "Forecasting copper", "authors": ["J. Doe"], "year": 2024}


# --------------------------------------------------------------------------- prompts
def test_prompt_loading_and_hash_stable() -> None:
    assert "papercard_v1" in available_versions()
    assert "evidence_location" in load_prompt("papercard_v1")
    assert prompt_hash("papercard_v1") == prompt_hash("papercard_v1")


def test_unknown_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nope_v9")


def test_prompt_embeds_schema_skeleton() -> None:
    # Weak models need to SEE the shape, not have it described. The skeleton is
    # derived from LLMPaperCardContent at render time so it can never drift.
    p = render_prompt(
        "papercard_v1",
        title="T",
        authors=["A"],
        year=2024,
        body="body {with braces}",
        low_confidence=False,
        max_chars=1000,
    )
    for name in LLMPaperCardContent.model_fields:
        assert f'"{name}"' in p, f"schema field {name} missing from rendered prompt"
    assert "(required)" in p
    assert "one of: empirical" in p  # enum values expanded
    assert "body {with braces}" in p  # substituted values are never format-rescanned
    assert "{schema}" not in p and "{{" not in p


def test_prompt_hash_folds_schema_into_version() -> None:
    # extraction_version must change when the schema skeleton changes, even if the
    # template file is untouched (re-extraction backfills key on this).
    base = prompt_hash("papercard_v1")
    with_schema = prompt_hash("papercard_v1", "some-schema-content")
    assert base != with_schema
    assert with_schema == prompt_hash("papercard_v1", "some-schema-content")  # stable


async def test_extract_survives_weak_model_output_in_one_shot() -> None:
    # Reproduces, verbatim, the quirk cluster observed live from a local 7B model:
    # prose around a fenced block, null-for-empty fields, unknown extra keys, a
    # wrong-case enum, and a bare string where a list belongs. Must extract on the
    # FIRST call - no repair round-trip.
    messy = (
        "Here is the extracted JSON you asked for:\n"
        "```json\n"
        + json.dumps(
            {
                "problem_statement": "Agents lack compact world models.",
                "methodology": {
                    "approach_summary": "VAE + MDN-RNN world model with a small controller.",
                    "techniques": "MDN-RNN",
                    "evaluation_protocol": None,
                    "baselines": None,
                },
                "datasets": [],
                "key_results": [{"claim": "trains in dream", "evidence_location": None}],
                "limitations": None,
                "contributions": "world model framework",
                "paper_type": "Empirical",
                "novelty": "high",
                "self_confidence": 0.8,
            }
        )
        + "\n```\nHope this helps!"
    )
    llm = ScriptedLLM([messy])
    card = await extract_paper_card(
        identity=_identity(), body="x", parse_confidence=1.0, llm=llm, settings=ExtractionSettings()
    )
    assert len(llm.models) == 1, "weak-model quirks must not burn a repair attempt"
    assert card.problem_statement.startswith("Agents lack")
    assert card.methodology.techniques == ["MDN-RNN"]
    assert card.methodology.baselines == []
    assert card.contributions == ["world model framework"]
    assert str(card.paper_type) == "empirical"
    # Unevidenced result was dropped by the hallucination guard, not crashed on.
    assert card.key_results == []


# --------------------------------------------------------------------------- scoring
def test_completeness_and_confidence() -> None:
    content = LLMPaperCardContent.model_validate(_content())
    assert completeness(content) == 1.0
    assert score_confidence(content, parse_confidence=1.0) > 0.9


# --------------------------------------------------------------------------- extraction
async def test_extract_happy_path() -> None:
    llm = ScriptedLLM([json.dumps(_content())])
    card = await extract_paper_card(
        identity=_identity(),
        body="full text",
        parse_confidence=1.0,
        llm=llm,
        settings=ExtractionSettings(),
    )
    assert card.title == "Forecasting copper"
    assert card.paper_id == "p1"
    assert card.confidence > 0.85
    assert not card.needs_review
    assert "papercard_v1@" in card.extraction_version
    assert card.key_results[0].evidence_location == "Table 3"
    assert llm.models == ["claude-haiku-4-5-20251001"]  # no escalation


async def test_extract_repairs_invalid_json() -> None:
    llm = ScriptedLLM(["this is not json", json.dumps(_content())])
    card = await extract_paper_card(
        identity=_identity(),
        body="x",
        parse_confidence=1.0,
        llm=llm,
        settings=ExtractionSettings(),
    )
    assert card.problem_statement.startswith("Forecasting")
    assert len(llm.models) == 2  # one repair


async def test_extract_gives_up_after_max_repairs() -> None:
    settings = ExtractionSettings(max_repair_attempts=1)
    llm = ScriptedLLM(["bad", "still bad"])
    with pytest.raises(SchemaValidationError):
        await extract_paper_card(
            identity=_identity(), body="x", parse_confidence=1.0, llm=llm, settings=settings
        )


async def test_extract_escalates_on_low_confidence() -> None:
    weak = _content(self_confidence=0.2, key_results=[], contributions=[], datasets=[])
    strong = _content(self_confidence=0.95)
    llm = ScriptedLLM([json.dumps(weak), json.dumps(strong)])
    card = await extract_paper_card(
        identity=_identity(),
        body="x",
        parse_confidence=1.0,
        llm=llm,
        settings=ExtractionSettings(),
    )
    assert llm.models == ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]
    assert card.extraction_model == "claude-sonnet-4-6"
    assert card.confidence > 0.85


async def test_extract_drops_unevidenced_results() -> None:
    c = _content(
        key_results=[
            {"claim": "good", "evidence_location": "Table 1"},
            {"claim": "unsupported", "evidence_location": ""},
        ]
    )
    llm = ScriptedLLM([json.dumps(c)])
    card = await extract_paper_card(
        identity=_identity(), body="x", parse_confidence=1.0, llm=llm, settings=ExtractionSettings()
    )
    assert len(card.key_results) == 1
    assert any("dropped_1" in r for r in card.review_reasons)


async def test_extract_flags_review_when_confidence_stays_low() -> None:
    weak = _content(
        self_confidence=0.1,
        key_results=[],
        contributions=[],
        datasets=[],
        problem_statement="",
        research_questions=[],
    )
    llm = ScriptedLLM([json.dumps(weak), json.dumps(weak)])
    card = await extract_paper_card(
        identity=_identity(), body="x", parse_confidence=0.4, llm=llm, settings=ExtractionSettings()
    )
    assert card.needs_review
    assert card.confidence < ExtractionSettings().review_confidence


async def test_extract_tracks_cost() -> None:
    tracker = CostTracker(per_job_cap=10.0, daily_cap=100.0)
    llm = ScriptedLLM([json.dumps(_content())])
    await extract_paper_card(
        identity=_identity(),
        body="x",
        parse_confidence=1.0,
        llm=llm,
        settings=ExtractionSettings(),
        cost=tracker,
    )
    assert tracker.job_usage.cost_usd > 0
    assert tracker.job_usage.calls == 1


class OutageThenScriptedLLM:
    """Raises a transport error for the primary model; succeeds on any other."""

    def __init__(self, good: str, failing_model: str) -> None:
        self._good = good
        self._failing = failing_model
        self.models: list[str] = []

    async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.models.append(model)
        if model == self._failing:
            raise RuntimeError("provider is down")
        return LLMResponse(text=self._good, input_tokens=500, output_tokens=200, model=model)


async def test_extract_falls_back_to_other_provider_on_outage() -> None:
    settings = ExtractionSettings()
    llm = OutageThenScriptedLLM(json.dumps(_content()), failing_model=settings.primary_model)
    card = await extract_paper_card(
        identity=_identity(), body="x", parse_confidence=1.0, llm=llm, settings=settings
    )
    assert card.extraction_model == settings.fallback_model
    assert llm.models[0] == settings.primary_model  # tried the primary first
    assert settings.fallback_model in llm.models


async def test_extract_schema_failure_does_not_trigger_fallback() -> None:
    # A prompt-shape problem gets the repair loop, never a provider switch.
    settings = ExtractionSettings(max_repair_attempts=1)
    llm = ScriptedLLM(["not json", "still not json"])
    with pytest.raises(SchemaValidationError):
        await extract_paper_card(
            identity=_identity(), body="x", parse_confidence=1.0, llm=llm, settings=settings
        )
    assert all(m == settings.primary_model for m in llm.models)


class EscalationOutageLLM:
    """Valid (weak) content from the primary; transport error from the escalation."""

    def __init__(self, weak: str, escalation_model: str) -> None:
        self._weak = weak
        self._escalation = escalation_model
        self.models: list[str] = []

    async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.models.append(model)
        if model == self._escalation:
            raise RuntimeError("provider is down")
        return LLMResponse(text=self._weak, input_tokens=500, output_tokens=200, model=model)


async def test_escalation_outage_keeps_primary_content() -> None:
    # Escalation improves already-valid content; its failure must degrade to the
    # primary extraction (flagged for review), not fail the whole job.
    settings = ExtractionSettings()
    weak = _content(self_confidence=0.1, key_results=[], contributions=[], datasets=[])
    llm = EscalationOutageLLM(json.dumps(weak), settings.escalation_model)
    card = await extract_paper_card(
        identity=_identity(), body="x", parse_confidence=1.0, llm=llm, settings=settings
    )
    assert card.extraction_model == settings.primary_model  # primary content kept
    assert card.confidence < settings.escalation_confidence  # still honestly low
    assert settings.escalation_model in llm.models  # escalation was attempted
