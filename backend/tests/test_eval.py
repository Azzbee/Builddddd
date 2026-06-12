from __future__ import annotations

import pytest
from lattice.eval.edge_quality import JudgedPair, evaluate_edges
from lattice.eval.extraction_eval import (
    evaluate_card,
    evaluate_corpus,
    regression_check,
)
from lattice.eval.golden import load_gold_cards, load_qa_pairs
from lattice.eval.metrics import (
    fuzzy_set_prf,
    pearson,
    precision_at_k,
    set_prf,
    token_f1,
)
from lattice.eval.retrieval_eval import (
    QAPair,
    citation_correctness,
    evaluate_rag,
    faithfulness,
)
from lattice.extraction.schemas import DatasetRef, Methodology, PaperCard, Result
from lattice.rag.models import AgentResult, Citation


# --------------------------------------------------------------------------- metrics
def test_set_prf() -> None:
    prf = set_prf({"LSTM", "VAR"}, {"lstm", "garch"})
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 1
    assert set_prf(set(), set()).f1 == 1.0


def test_fuzzy_set_prf_matches_near_strings() -> None:
    prf = fuzzy_set_prf(["LSTM with attention"], ["attention based lstm"], threshold=0.6)
    assert prf.tp == 1


def test_precision_at_k_and_pearson_and_token_f1() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 1], [1, 1]) == 0.0
    assert token_f1("the cat sat", "the cat sat") == 1.0


# --------------------------------------------------------------------------- extraction eval
def _pred_card() -> PaperCard:
    return PaperCard(
        paper_id="golden-001",
        title="Copper",
        problem_statement="Forecasting daily LME copper prices is hard due to nonstationarity.",
        methodology=Methodology(approach_summary="LSTM", techniques=["LSTM", "attention", "ARIMA"]),
        methods_taxonomy=["LSTM"],
        datasets=[DatasetRef(name="LME copper")],
        key_results=[
            Result(claim="LSTM with attention beats ARIMA on RMSE", evidence_location="T1"),
        ],
        limitations=["evaluated on a single commodity"],
    )


def test_evaluate_card_per_field() -> None:
    gold = load_gold_cards()[0]
    per = evaluate_card(_pred_card(), gold)
    assert per["methods"].f1 > 0.5
    assert per["datasets"].f1 == 1.0
    assert per["problem"].tp == 1


def test_evaluate_corpus_and_regression_check() -> None:
    golds = load_gold_cards()
    report = evaluate_corpus({"golden-001": _pred_card()}, golds)
    assert report.n_papers == 1
    assert 0.0 <= report.macro_f1 <= 1.0
    assert regression_check(0.80, 0.82) is True  # 2pt drop ok
    assert regression_check(0.70, 0.82) is False  # 12pt drop fails


# --------------------------------------------------------------------------- retrieval eval
def _result(cite_ids: list[str], abstained: bool = False) -> AgentResult:
    return AgentResult(
        answer="LSTM with attention beat ARIMA on RMSE.",
        citations=[Citation(i + 1, pid, "T") for i, pid in enumerate(cite_ids)],
        abstained=abstained,
    )


def test_citation_correctness_and_faithfulness() -> None:
    res = _result(["golden-001"])
    assert citation_correctness(res, {"golden-001"}) == 1.0
    assert citation_correctness(_result(["wrong"]), {"golden-001"}) == 0.0
    assert faithfulness(res) == 1.0
    assert faithfulness(_result([], abstained=True)) == 1.0  # abstention is faithful
    assert faithfulness(_result([])) == 0.0  # asserted with no grounding


def test_evaluate_rag_aggregate_and_thresholds() -> None:
    pairs = load_qa_pairs()
    results = [_result(["golden-001"]), _result(["golden-002"])]
    report = evaluate_rag(pairs, results)
    assert report.citation_correctness == 1.0
    assert report.faithfulness >= 0.85
    assert report.passes()
    assert report.n == 2


def test_evaluate_rag_alignment_check() -> None:
    with pytest.raises(ValueError):
        evaluate_rag([QAPair("q", "a")], [])


# --------------------------------------------------------------------------- edge quality
def test_evaluate_edges_correlation_and_pak() -> None:
    pairs = [
        JudgedPair("p1", "a", 0.9, 0.9),
        JudgedPair("p1", "b", 0.7, 0.7),
        JudgedPair("p1", "c", 0.2, 0.1),
    ]
    report = evaluate_edges(pairs, k=2)
    assert report.correlation > 0.9
    assert report.precision_at_5 > 0.0
    assert report.n_pairs == 3


def test_evaluate_edges_empty() -> None:
    assert evaluate_edges([]).n_pairs == 0


def test_golden_loaders() -> None:
    assert len(load_gold_cards()) >= 2
    assert len(load_qa_pairs()) >= 2
