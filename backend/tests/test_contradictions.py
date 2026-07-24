from __future__ import annotations

import json

import pytest
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.graph.contradictions import (
    ClaimRecord,
    ClaimRelation,
    HeuristicNLIJudge,
    LLMNLIJudge,
    candidate_pairs,
    detect_relations,
)


def _claim(cid: str, paper: str, text: str, concept: str = "forecasting") -> ClaimRecord:
    return ClaimRecord(claim_id=cid, paper_id=paper, text=text, concept=concept)


async def test_heuristic_detects_contradiction() -> None:
    judge = HeuristicNLIJudge()
    rel, conf = await judge.judge(
        "LSTM models significantly improve copper price forecasting accuracy",
        "LSTM models show no significant improvement in copper price forecasting",
    )
    assert rel == ClaimRelation.CONTRADICTS
    assert conf > 0.3


async def test_heuristic_detects_support() -> None:
    judge = HeuristicNLIJudge()
    rel, _ = await judge.judge(
        "Attention improves forecasting accuracy on copper prices",
        "Attention improves forecasting accuracy for copper price series",
    )
    assert rel == ClaimRelation.SUPPORTS


async def test_heuristic_no_false_contradiction_on_generic_overlap() -> None:
    # Different comparisons that merely share generic tokens (forecasting/accuracy/
    # lstm) must NOT be called a contradiction - this false positive used to suppress
    # legitimate known-knowns downstream.
    judge = HeuristicNLIJudge()
    rel, _ = await judge.judge(
        "LSTM with attention significantly improves forecasting accuracy over ARIMA",
        "Transformers show no significant improvement in forecasting accuracy over LSTM baselines",
    )
    assert rel == ClaimRelation.UNRELATED


async def test_heuristic_unrelated_when_different_subject() -> None:
    judge = HeuristicNLIJudge()
    rel, _ = await judge.judge(
        "LSTM improves copper price forecasting",
        "Reinforcement learning improves robot locomotion control",
    )
    assert rel == ClaimRelation.UNRELATED


def test_candidate_pairs_same_concept_diff_paper() -> None:
    claims = [
        _claim("c1", "p1", "a", "forecasting"),
        _claim("c2", "p2", "b", "forecasting"),
        _claim("c3", "p1", "c", "forecasting"),  # same paper as c1 -> skip pair (c1,c3)
        _claim("c4", "p3", "d", "robotics"),  # different concept -> no cross pairs
    ]
    pairs = candidate_pairs(claims)
    keyset = {(a.claim_id, b.claim_id) for a, b in pairs}
    assert ("c1", "c2") in keyset
    assert ("c2", "c3") in keyset
    assert ("c1", "c3") not in keyset  # same paper
    assert all("c4" not in k for k in keyset)  # isolated concept


async def test_detect_relations_collects_contradictions() -> None:
    claims = [
        _claim("c1", "p1", "Transformers significantly improve forecasting accuracy"),
        _claim("c2", "p2", "Transformers show no significant improvement in forecasting accuracy"),
        _claim("c3", "p3", "Transformers improve forecasting accuracy substantially"),
    ]
    report = await detect_relations(claims, HeuristicNLIJudge(), min_confidence=0.3)
    assert len(report.contradictions) >= 1
    edge = report.contradictions[0]
    assert edge.source_paper != edge.target_paper
    assert edge.to_json()["relation"] == "CONTRADICTS"


async def test_llm_judge_parses_json() -> None:
    class FakeLLM:
        async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
            return LLMResponse(
                text=json.dumps({"relation": "CONTRADICTS", "confidence": 0.9}),
                model=model,
            )

    rel, conf = await LLMNLIJudge(FakeLLM()).judge("a", "b")
    assert rel == ClaimRelation.CONTRADICTS and conf == 0.9


async def test_llm_judge_handles_bad_output() -> None:
    class BadLLM:
        async def complete(self, model, messages, **kwargs) -> LLMResponse:
            return LLMResponse(text="not json", model=model)

    rel, conf = await LLMNLIJudge(BadLLM()).judge("a", "b")
    assert rel == ClaimRelation.UNRELATED and conf == 0.0


@pytest.mark.parametrize("relation", ["SUPPORTS", "CONTRADICTS", "EXTENDS"])
async def test_writer_persists_claim_relation(relation: str) -> None:
    from lattice.graph.store import FakeGraphStore
    from lattice.graph.writer import GraphWriter

    store = FakeGraphStore()
    writer = GraphWriter(store, "ws1")
    await writer.upsert_claim_relation("c1", "c2", relation, 0.8)
    q, params = store.calls[-1]
    assert relation in q and params["confidence"] == 0.8


async def test_writer_rejects_unknown_relation() -> None:
    from lattice.graph.store import FakeGraphStore
    from lattice.graph.writer import GraphWriter

    writer = GraphWriter(FakeGraphStore(), "ws1")
    with pytest.raises(ValueError):
        await writer.upsert_claim_relation("c1", "c2", "DROP", 0.8)


# ------------------------------------------------------------- incremental path
def test_candidate_pairs_for_only_pairs_new_against_existing() -> None:
    from lattice.graph.contradictions import candidate_pairs_for

    new = [_claim("n1", "pNew", "LSTM improves accuracy", "forecasting")]
    existing = [
        _claim("e1", "pOld", "LSTM shows no improvement", "forecasting"),
        _claim("e2", "pOld", "GRU is fast", "architecture"),  # different concept
        _claim("e3", "pNew", "own claim", "forecasting"),  # same paper: excluded
    ]
    pairs = candidate_pairs_for(new, existing)
    assert [(a.claim_id, b.claim_id) for a, b in pairs] == [("n1", "e1")]


async def test_detect_relations_for_flags_contradiction() -> None:
    from lattice.graph.contradictions import detect_relations_for

    new = [
        _claim(
            "n1",
            "pNew",
            "LSTM significantly improves forecasting accuracy over ARIMA",
            "forecasting",
        )
    ]
    existing = [
        _claim(
            "e1",
            "pOld",
            "LSTM shows no improvement in forecasting accuracy over ARIMA",
            "forecasting",
        )
    ]
    report = await detect_relations_for(new, existing, HeuristicNLIJudge())
    assert len(report.contradictions) == 1
    edge = report.contradictions[0]
    assert edge.source_paper == "pNew" and edge.target_paper == "pOld"
