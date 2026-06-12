"""Edge-quality eval: does the composite weight track human-judged relatedness?

For sampled paper pairs across the weight spectrum, compare ``w(i,j)`` to a judged
relatedness score and report correlation; report precision@k for "most related"
lists. This makes the weight coefficients (alpha,beta,gamma,delta) tunable as a
measurable decision rather than vibes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice.eval.metrics import pearson, precision_at_k


@dataclass
class JudgedPair:
    source_id: str
    target_id: str
    computed_weight: float
    judged_relatedness: float  # 0-1 from an LLM judge or human


@dataclass
class EdgeQualityReport:
    correlation: float
    precision_at_5: float
    n_pairs: int
    per_source_p_at_k: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "correlation": round(self.correlation, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "n_pairs": self.n_pairs,
        }


def evaluate_edges(
    pairs: list[JudgedPair], *, relevance_threshold: float = 0.6, k: int = 5
) -> EdgeQualityReport:
    """Correlation between computed weights and judged relatedness, plus P@k."""
    if not pairs:
        return EdgeQualityReport(0.0, 0.0, 0)

    corr = pearson([p.computed_weight for p in pairs], [p.judged_relatedness for p in pairs])

    # Group by source for ranking metrics.
    by_source: dict[str, list[JudgedPair]] = {}
    for p in pairs:
        by_source.setdefault(p.source_id, []).append(p)

    per_source: dict[str, float] = {}
    for src, items in by_source.items():
        ranked = sorted(items, key=lambda x: x.computed_weight, reverse=True)
        relevant = {x.target_id for x in items if x.judged_relatedness >= relevance_threshold}
        per_source[src] = precision_at_k([x.target_id for x in ranked], relevant, k)

    p_at_k = sum(per_source.values()) / len(per_source) if per_source else 0.0
    return EdgeQualityReport(
        correlation=corr,
        precision_at_5=p_at_k,
        n_pairs=len(pairs),
        per_source_p_at_k=per_source,
    )
