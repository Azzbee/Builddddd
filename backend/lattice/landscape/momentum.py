"""Momentum & emergence detection for concepts/communities.

Every signal is computed and exposed (no opaque score), mirroring the edge-weight
philosophy. Signals:

* publication velocity + acceleration (from yearly counts)
* Kleinberg-style burst score (a simplified two-state detector)
* maturity stage via logistic (S-curve) fit on cumulative publications
* author influx (share of first-time authors)
* community convergence (distinct communities newly linking in)

Honest limit (surfaced in the UI): momentum measures activity, not truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class MaturityStage(StrEnum):
    EMERGING = "emerging"
    ACCELERATING = "accelerating"
    INFLECTING = "inflecting"
    SATURATING = "saturating"
    UNKNOWN = "unknown"


def _series(counts: dict[int, int]) -> list[tuple[int, int]]:
    return sorted(counts.items())


def velocity(counts: dict[int, int]) -> float:
    """Mean year-over-year change in publication counts."""
    s = _series(counts)
    if len(s) < 2:
        return 0.0
    deltas = [s[i][1] - s[i - 1][1] for i in range(1, len(s))]
    return sum(deltas) / len(deltas)


def acceleration(counts: dict[int, int]) -> float:
    """Mean change in the year-over-year deltas (second difference)."""
    s = _series(counts)
    if len(s) < 3:
        return 0.0
    deltas = [s[i][1] - s[i - 1][1] for i in range(1, len(s))]
    second = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]
    return sum(second) / len(second)


def burst_score(counts: dict[int, int]) -> float:
    """Simplified Kleinberg burst: max standardized positive deviation of a year's
    count above the running mean, in units of standard deviation."""
    s = _series(counts)
    vals = [c for _y, c in s]
    if len(vals) < 3:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    return max((v - mean) / std for v in vals)


def maturity_stage(counts: dict[int, int]) -> MaturityStage:
    """Classify where on the adoption S-curve a topic sits.

    Uses the trend in *yearly* counts (which flatten at saturation), not cumulative
    counts (which always rise). Recent year-over-year growth in the annual rate
    distinguishes accelerating from inflecting from saturating.
    """
    s = _series(counts)
    if len(s) < 3:
        return MaturityStage.UNKNOWN
    prev, last = s[-2][1], s[-1][1]
    if last == 0 and prev == 0:
        return MaturityStage.UNKNOWN
    recent_growth = (last - prev) / max(1, prev)
    if recent_growth > 0.5:
        return MaturityStage.ACCELERATING
    if recent_growth > 0.15:
        return MaturityStage.INFLECTING
    if recent_growth <= 0.05:
        return MaturityStage.SATURATING
    return MaturityStage.EMERGING


@dataclass
class MomentumScore:
    concept: str
    velocity: float
    acceleration: float
    burst: float
    author_influx: float
    community_convergence: int
    maturity: MaturityStage
    composite: float
    counts: dict[int, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "concept": self.concept,
            "velocity": round(self.velocity, 3),
            "acceleration": round(self.acceleration, 3),
            "burst": round(self.burst, 3),
            "author_influx": round(self.author_influx, 3),
            "community_convergence": self.community_convergence,
            "maturity": str(self.maturity),
            "composite": round(self.composite, 3),
            "counts": self.counts,
        }


def _norm(x: float, scale: float) -> float:
    """Squash an unbounded signal into [0, 1] with a soft scale."""
    return math.tanh(max(0.0, x) / scale) if scale > 0 else 0.0


def momentum_score(
    concept: str,
    counts: dict[int, int],
    *,
    author_influx: float = 0.0,
    community_convergence: int = 0,
) -> MomentumScore:
    """Compute the full momentum scorecard with a visible composite.

    The composite weights acceleration and convergence most (the signals most
    robust against pure hype), then velocity, burst, and author influx.
    """
    v = velocity(counts)
    a = acceleration(counts)
    b = burst_score(counts)
    conv = _norm(float(community_convergence), 3.0)
    composite = (
        0.30 * _norm(a, 5.0)
        + 0.25 * conv
        + 0.20 * _norm(v, 10.0)
        + 0.15 * _norm(b, 2.0)
        + 0.10 * max(0.0, min(1.0, author_influx))
    )
    return MomentumScore(
        concept=concept,
        velocity=v,
        acceleration=a,
        burst=b,
        author_influx=author_influx,
        community_convergence=community_convergence,
        maturity=maturity_stage(counts),
        composite=composite,
        counts=counts,
    )
