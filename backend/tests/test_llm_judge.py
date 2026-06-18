"""Tests for the LLM-as-judge RAG eval harness (offline, scripted model)."""

from __future__ import annotations

import json

import pytest
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.eval.llm_judge import (
    JudgedRagReport,
    LLMJudge,
    evaluate_rag_judged,
)
from lattice.eval.retrieval_eval import QAPair
from lattice.rag.models import AgentResult, Citation


class FakeJudgeLLM:
    """Routes by prompt content so it is robust to judge call ordering.

    ``supported`` controls whether faithfulness claims are marked grounded;
    ``score`` is returned for the scalar dimensions; ``precision`` controls the
    per-context relevance verdicts.
    """

    def __init__(self, *, supported: bool = True, score: float = 0.9, precision: bool = True):
        self.supported = supported
        self.score = score
        self.precision = precision
        self.calls = 0

    async def complete(self, model: str, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1].content
        if "atomic factual claims" in prompt:
            payload = {"claims": [{"claim": "c1", "supported": self.supported},
                                  {"claim": "c2", "supported": True}]}
        elif "directly the ANSWER addresses" in prompt:
            payload = {"score": self.score, "reasoning": "ok"}
        elif "numbered CONTEXT" in prompt:
            payload = {"verdicts": [self.precision, True]}
        elif "reference GOLD" in prompt:
            payload = {"score": self.score, "reasoning": "matches"}
        else:
            payload = {}
        return LLMResponse(text=json.dumps(payload), model=model)


def _result(answer: str = "LSTM beat ARIMA on RMSE.", cites: list[str] | None = None,
            abstained: bool = False) -> AgentResult:
    cites = cites or ["golden-001"]
    return AgentResult(
        answer=answer,
        citations=[Citation(i + 1, pid, "T", snippet="evidence snippet") for i, pid in enumerate(cites)],
        abstained=abstained,
    )


# --------------------------------------------------------------------------- judge units
@pytest.mark.asyncio
async def test_faithfulness_fraction_supported() -> None:
    judge = LLMJudge(FakeJudgeLLM(supported=False))
    v = await judge.faithfulness("answer", ["ctx"])
    assert v.score == 0.5  # one of two claims supported
    assert v.details["supported"] == 1


@pytest.mark.asyncio
async def test_faithfulness_empty_answer_is_zero() -> None:
    judge = LLMJudge(FakeJudgeLLM())
    assert (await judge.faithfulness("   ", ["ctx"])).score == 0.0


@pytest.mark.asyncio
async def test_faithfulness_no_claims_is_one() -> None:
    class NoClaims(FakeJudgeLLM):
        async def complete(self, model: str, messages: list[LLMMessage], **kw: object) -> LLMResponse:
            return LLMResponse(text='{"claims": []}', model=model)

    judge = LLMJudge(NoClaims())
    v = await judge.faithfulness("answer with no facts", ["ctx"])
    assert v.score == 1.0 and v.details["n_claims"] == 0


@pytest.mark.asyncio
async def test_relevance_and_correctness_clamped() -> None:
    judge = LLMJudge(FakeJudgeLLM(score=1.5))  # out of range -> clamped to 1.0
    assert (await judge.answer_relevance("q", "a")).score == 1.0
    assert (await judge.answer_correctness("a", "gold")).score == 1.0


@pytest.mark.asyncio
async def test_context_precision_fraction_and_empty() -> None:
    judge = LLMJudge(FakeJudgeLLM(precision=False))
    v = await judge.context_precision("q", ["c1", "c2"])
    assert v.score == 0.5  # [False, True]
    assert (await judge.context_precision("q", [])).score == 0.0


@pytest.mark.asyncio
async def test_malformed_json_degrades_gracefully() -> None:
    class Garbage(FakeJudgeLLM):
        async def complete(self, model: str, messages: list[LLMMessage], **kw: object) -> LLMResponse:
            return LLMResponse(text="not json at all", model=model)

    judge = LLMJudge(Garbage())
    assert (await judge.answer_relevance("q", "a")).score == 0.0


# --------------------------------------------------------------------------- harness
@pytest.mark.asyncio
async def test_evaluate_rag_judged_aggregates_and_passes() -> None:
    pairs = [QAPair("q1", "gold1", {"golden-001"}), QAPair("q2", "gold2", {"golden-002"})]
    results = [_result(cites=["golden-001"]), _result(cites=["golden-002"])]
    report = await evaluate_rag_judged(pairs, results, LLMJudge(FakeJudgeLLM()))
    assert isinstance(report, JudgedRagReport)
    assert report.n == 2
    assert report.faithfulness == 1.0  # both claims supported
    assert report.citation_correctness == 1.0
    assert report.answer_correctness == pytest.approx(0.9)
    assert report.passes()


@pytest.mark.asyncio
async def test_abstention_scored_without_model_call() -> None:
    fake = FakeJudgeLLM()
    pairs = [QAPair("q", "gold", {"p1"})]
    results = [_result(answer="I don't know.", cites=[], abstained=True)]
    report = await evaluate_rag_judged(pairs, results, LLMJudge(fake))
    assert fake.calls == 0  # no LLM calls for an abstention
    assert report.abstention_rate == 1.0
    assert report.faithfulness == 1.0  # faithful by construction
    assert report.answer_correctness == 0.0


@pytest.mark.asyncio
async def test_explicit_contexts_used_for_precision() -> None:
    fake = FakeJudgeLLM(precision=False)
    pairs = [QAPair("q", "gold", {"golden-001"})]
    results = [_result()]
    report = await evaluate_rag_judged(
        pairs, results, LLMJudge(fake), contexts=[["snippet a", "snippet b"]]
    )
    assert report.context_precision == 0.5


@pytest.mark.asyncio
async def test_alignment_errors() -> None:
    judge = LLMJudge(FakeJudgeLLM())
    with pytest.raises(ValueError):
        await evaluate_rag_judged([QAPair("q", "a")], [], judge)
    with pytest.raises(ValueError):
        await evaluate_rag_judged([QAPair("q", "a")], [_result()], judge, contexts=[])


@pytest.mark.asyncio
async def test_empty_set_returns_zero_report() -> None:
    report = await evaluate_rag_judged([], [], LLMJudge(FakeJudgeLLM()))
    assert report.n == 0 and report.to_json()["n"] == 0
