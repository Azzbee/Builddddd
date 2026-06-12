#!/usr/bin/env python3
"""Run the evaluation harnesses and print reports.

In ``--ci`` mode everything is deterministic and offline: it exercises the
extraction, retrieval, and edge-quality harnesses over the golden set and a
synthetic edge sample, and enforces the PRD thresholds. This guarantees the eval
code paths stay green in CI without LLM keys or external services.

    cd backend && uv run python ../scripts/run_eval.py --ci
"""

from __future__ import annotations

import json
import sys

from lattice.eval.edge_quality import JudgedPair, evaluate_edges
from lattice.eval.extraction_eval import evaluate_corpus, regression_check
from lattice.eval.golden import load_gold_cards, load_qa_pairs
from lattice.eval.retrieval_eval import evaluate_rag
from lattice.extraction.schemas import DatasetRef, Methodology, PaperCard, Result
from lattice.rag.models import AgentResult, Citation


def _card_from_gold(gold) -> PaperCard:  # type: ignore[no-untyped-def]
    """Build a PaperCard that matches a gold annotation (harness sanity check)."""
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


def main(argv: list[str]) -> int:
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
    print(json.dumps(report, indent=2))

    if "--ci" in argv:
        # Harness sanity: gold-matched predictions must score perfectly, the RAG
        # thresholds must pass, and weight/relatedness must correlate positively.
        ok = (
            regression_check(extraction.macro_f1, 1.0)
            and extraction.macro_f1 >= 0.99
            and retrieval.passes()
            and edge_report.correlation > 0.5
        )
        if not ok:
            print("EVAL CI CHECK FAILED", file=sys.stderr)
            return 1
        print("\nEVAL CI CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
