"""Extraction eval: field-level precision/recall/F1 vs a golden annotation set.

Reports per-field PRF (problem, methods, datasets, results, limitations) and a
macro F1. CI fails if macro F1 regresses more than a configured number of points
against the recorded baseline for a prompt version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice.eval.metrics import PRF, _prf_from_counts, fuzzy_set_prf, macro_f1, set_prf
from lattice.extraction.schemas import PaperCard

SET_FIELDS = ("methods", "datasets")
TEXT_FIELDS = ("problem", "results", "limitations")


@dataclass
class GoldCard:
    paper_id: str
    problem: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def _predicted_fields(card: PaperCard) -> dict[str, list[str]]:
    return {
        "problem": [card.problem_statement] if card.problem_statement else [],
        "methods": sorted(card.normalized_methods),
        "datasets": [d.name for d in card.datasets],
        "results": [r.claim for r in card.key_results],
        "limitations": list(card.limitations),
    }


def evaluate_card(pred: PaperCard, gold: GoldCard) -> dict[str, PRF]:
    p = _predicted_fields(pred)
    out: dict[str, PRF] = {}
    out["methods"] = set_prf(set(p["methods"]), set(gold.methods))
    out["datasets"] = set_prf(set(p["datasets"]), set(gold.datasets))
    out["problem"] = fuzzy_set_prf(p["problem"], gold.problem)
    out["results"] = fuzzy_set_prf(p["results"], gold.results)
    out["limitations"] = fuzzy_set_prf(p["limitations"], gold.limitations)
    return out


@dataclass
class ExtractionReport:
    per_field: dict[str, PRF]
    macro_f1: float
    n_papers: int

    def to_json(self) -> dict[str, object]:
        return {
            "per_field": {k: v.to_json() for k, v in self.per_field.items()},
            "macro_f1": round(self.macro_f1, 4),
            "n_papers": self.n_papers,
        }


def evaluate_corpus(
    predictions: dict[str, PaperCard], golds: list[GoldCard]
) -> ExtractionReport:
    """Aggregate micro PRF per field across all papers (summing TP/FP/FN)."""
    fields = ["problem", "methods", "datasets", "results", "limitations"]
    tp = dict.fromkeys(fields, 0)
    fp = dict.fromkeys(fields, 0)
    fn = dict.fromkeys(fields, 0)
    n = 0
    for gold in golds:
        pred = predictions.get(gold.paper_id)
        if pred is None:
            continue
        n += 1
        per = evaluate_card(pred, gold)
        for f in fields:
            tp[f] += per[f].tp
            fp[f] += per[f].fp
            fn[f] += per[f].fn
    per_field = {f: _prf_from_counts(tp[f], fp[f], fn[f]) for f in fields}
    return ExtractionReport(per_field=per_field, macro_f1=macro_f1(per_field), n_papers=n)


def regression_check(current_f1: float, baseline_f1: float, max_drop_points: float = 5.0) -> bool:
    """Return True if extraction quality is acceptable (no >max_drop regression)."""
    return (baseline_f1 - current_f1) * 100.0 <= max_drop_points
