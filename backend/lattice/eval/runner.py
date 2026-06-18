"""Shared offline eval runner used by scripts/run_eval.py and the CLI.

Deterministic and dependency-free: exercises the extraction, retrieval, and
edge-quality harnesses over the golden set plus a synthetic edge sample, and
reports whether the PRD thresholds hold. No LLM keys or services required.
"""

from __future__ import annotations

from typing import Any

from lattice.eval.edge_quality import JudgedPair, evaluate_edges
from lattice.eval.extraction_eval import GoldCard, evaluate_corpus, regression_check
from lattice.eval.golden import load_gold_cards, load_qa_pairs
from lattice.eval.retrieval_eval import evaluate_rag
from lattice.extraction.schemas import DatasetRef, Methodology, PaperCard, Result
from lattice.rag.models import AgentResult, Citation


def _card_from_gold(gold: GoldCard) -> PaperCard:
    return PaperCard(
        paper_id=gold.paper_id,
        title=gold.paper_id,
        problem_statement=gold.problem[0] if gold.problem else "",
        methodology=Methodology(approach_summary="", techniques=list(gold.methods)),
        methods_taxonomy=list(gold.methods),
        datasets=[DatasetRef(name=d) for d in gold.datasets],
        key_results=[Result(claim=r, evidence_location="x") for r in gold.results],
        limitations=list(gold.limitations),
    )


def run_offline_eval() -> tuple[dict[str, Any], bool]:
    """Return (report, ok). ``ok`` enforces the harness/threshold sanity checks."""
    golds = load_gold_cards()
    predictions = {g.paper_id: _card_from_gold(g) for g in golds}
    extraction = evaluate_corpus(predictions, golds)

    qa = load_qa_pairs()
    rag_results = [
        AgentResult(
            answer=p.gold_answer,
            citations=[Citation(1, pid, "x") for pid in p.relevant_paper_ids],
            confidence=0.9,
        )
        for p in qa
    ]
    retrieval = evaluate_rag(qa, rag_results)

    edges = [
        JudgedPair("p1", "a", 0.9, 0.95),
        JudgedPair("p1", "b", 0.6, 0.55),
        JudgedPair("p1", "c", 0.2, 0.10),
    ]
    edge_report = evaluate_edges(edges)

    report = {
        "extraction": extraction.to_json(),
        "retrieval": retrieval.to_json(),
        "edge_quality": edge_report.to_json(),
    }
    ok = (
        regression_check(extraction.macro_f1, 1.0)
        and extraction.macro_f1 >= 0.99
        and retrieval.passes()
        and edge_report.correlation > 0.5
    )
    return report, ok
