"""Reusable evaluation metrics: precision/recall/F1, fuzzy set matching,
rank metrics, and correlation. Pure functions, no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rapidfuzz import fuzz

from lattice.core.hashing import normalize_text


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def to_json(self) -> dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def set_prf(predicted: set[str], gold: set[str]) -> PRF:
    """Exact set-based P/R/F1 over normalized strings."""
    pred = {normalize_text(p) for p in predicted if p.strip()}
    gold_n = {normalize_text(g) for g in gold if g.strip()}
    if not pred and not gold_n:
        return PRF(1.0, 1.0, 1.0)
    tp = len(pred & gold_n)
    return _prf_from_counts(tp, len(pred - gold_n), len(gold_n - pred))


def fuzzy_set_prf(predicted: list[str], gold: list[str], threshold: float = 0.8) -> PRF:
    """P/R/F1 where a predicted item matches a gold item if fuzzy score >= threshold.

    Greedy one-to-one matching. Used for free-text fields (problem, results,
    limitations) where exact string match is too strict.
    """
    pred = [normalize_text(p) for p in predicted if p.strip()]
    gold_n = [normalize_text(g) for g in gold if g.strip()]
    if not pred and not gold_n:
        return PRF(1.0, 1.0, 1.0)
    used: set[int] = set()
    tp = 0
    for p in pred:
        for i, g in enumerate(gold_n):
            if i in used:
                continue
            if fuzz.token_set_ratio(p, g) / 100.0 >= threshold:
                used.add(i)
                tp += 1
                break
    return _prf_from_counts(tp, len(pred) - tp, len(gold_n) - tp)


def macro_f1(prfs: dict[str, PRF]) -> float:
    if not prfs:
        return 0.0
    return sum(p.f1 for p in prfs.values()) / len(prfs)


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x in relevant) / len(top)


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 for degenerate input."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between two strings (a cheap answer-relevance proxy)."""
    pt = normalize_text(prediction).split()
    rt = normalize_text(reference).split()
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    common: dict[str, int] = {}
    rt_counts: dict[str, int] = {}
    for t in rt:
        rt_counts[t] = rt_counts.get(t, 0) + 1
    overlap = 0
    for t in pt:
        if common.get(t, 0) < rt_counts.get(t, 0):
            overlap += 1
            common[t] = common.get(t, 0) + 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(rt)
    return 2 * precision * recall / (precision + recall)
