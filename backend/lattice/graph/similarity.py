"""The composite similarity function: Lattice's core IP.

Edge weight between papers i, j:

    w(i,j) = sigma( a*S_sem + b*S_meth + c*S_cit + d*S_data ) * T(i,j)

Every component is normalized to [0,1] and reported alongside the final weight so
edges are *auditable* ("edge anatomy" in the UI, ``weight_components`` in the DB).

Design notes (see docs/SIMILARITY.md for the full story):

* S_sem uses SPECTER2 cosine, corpus-calibrated, because SPECTER-family models
  discriminate related pairs better than unrelated ones, so raw cosine is
  optimistic and must be rescaled against the corpus distribution.
* S_meth blends Jaccard over normalized method tags with cosine over the
  methodology-section embeddings; section-level comparison materially improves
  discrimination over title+abstract alone.
* S_cit takes the max of bibliographic coupling and co-citation, saturated to 1.0
  when a direct citation exists.
* S_data is Jaccard over resolved dataset nodes: small but high-precision.
* When a component's inputs are missing (e.g. citation data not yet enriched),
  the function degrades gracefully by renormalizing the coefficients over the
  components actually available, and records which those were.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from lattice.config import SimilarityWeights


# ---------------------------------------------------------------------------
# Vector / set primitives
# ---------------------------------------------------------------------------
def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine similarity in [-1, 1], or None if either vector is missing/zero."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets; 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def sigmoid(x: float, gain: float, midpoint: float) -> float:
    """Logistic squashing centered at ``midpoint`` with steepness ``gain``."""
    return 1.0 / (1.0 + math.exp(-gain * (x - midpoint)))


# ---------------------------------------------------------------------------
# Corpus calibration for S_sem
# ---------------------------------------------------------------------------
@dataclass
class CosineCalibrator:
    """Robust min-max rescaler fit on the corpus distribution of SPECTER2 cosines.

    Uses percentiles (default p5/p95) so a handful of near-duplicate or unrelated
    outliers don't dominate the scale. Until fit, it is an identity clamp on the
    typical [0,1] cosine range, which is a safe default.
    """

    lo: float = 0.0
    hi: float = 1.0
    fitted: bool = False

    @classmethod
    def fit(
        cls, cosines: list[float], low_pct: float = 5.0, high_pct: float = 95.0
    ) -> CosineCalibrator:
        vals = [c for c in cosines if c is not None]
        if len(vals) < 2:
            return cls(lo=0.0, hi=1.0, fitted=False)
        lo = float(np.percentile(vals, low_pct))
        hi = float(np.percentile(vals, high_pct))
        if hi <= lo:
            hi = lo + 1e-6
        return cls(lo=lo, hi=hi, fitted=True)

    def transform(self, raw: float) -> float:
        scaled = (raw - self.lo) / (self.hi - self.lo)
        return max(0.0, min(1.0, scaled))


# ---------------------------------------------------------------------------
# Paper feature representation
# ---------------------------------------------------------------------------
@dataclass
class PaperFeatures:
    """The minimal feature bundle the similarity function consumes for one paper.

    Decoupled from PaperCard so similarity stays pure and trivially testable, and
    so features can be hydrated from cache/DB without reconstructing full cards.
    """

    paper_id: str
    specter: np.ndarray | None = None
    methodology_embedding: np.ndarray | None = None
    methods: set[str] = field(default_factory=set)
    datasets: set[str] = field(default_factory=set)
    references: set[str] = field(default_factory=set)  # cited paper ids
    ingested_at: datetime | None = None


@dataclass
class WeightResult:
    """The output of the composite function: final weight plus full anatomy."""

    weight: float
    raw_score: float  # pre-sigmoid linear combination
    temporal_factor: float
    components: dict[str, float]  # component -> [0,1] value (available ones only)
    available: list[str]  # which components contributed
    coefficients: dict[str, float]  # effective (renormalized) coefficients used

    def to_json(self) -> dict[str, object]:
        return {
            "weight": round(self.weight, 6),
            "raw_score": round(self.raw_score, 6),
            "temporal_factor": round(self.temporal_factor, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "available": self.available,
            "coefficients": {k: round(v, 6) for k, v in self.coefficients.items()},
        }


# ---------------------------------------------------------------------------
# Individual components
# ---------------------------------------------------------------------------
def s_sem(i: PaperFeatures, j: PaperFeatures, calibrator: CosineCalibrator) -> float | None:
    raw = cosine(i.specter, j.specter)
    if raw is None:
        return None
    return calibrator.transform(raw)


def s_meth(i: PaperFeatures, j: PaperFeatures, section_weight: float) -> float | None:
    tag_sim = jaccard(i.methods, j.methods) if (i.methods or j.methods) else None
    sec_sim = cosine(i.methodology_embedding, j.methodology_embedding)
    if sec_sim is not None:
        sec_sim = max(0.0, sec_sim)  # treat anti-correlation as "not similar"

    if tag_sim is None and sec_sim is None:
        return None
    if sec_sim is None:
        return tag_sim
    if tag_sim is None:
        return sec_sim
    return section_weight * sec_sim + (1.0 - section_weight) * tag_sim


def s_cit(
    i: PaperFeatures,
    j: PaperFeatures,
    *,
    co_citation: float | None = None,
    direct_citation: bool = False,
) -> float | None:
    """Citation-structure similarity.

    max(bibliographic coupling, co-citation), saturated to 1.0 on a direct cite.
    Returns None only when no citation signal of any kind is available.
    """
    have_refs = bool(i.references or j.references)
    if not have_refs and co_citation is None and not direct_citation:
        return None
    coupling = jaccard(i.references, j.references) if have_refs else 0.0
    base = max(coupling, co_citation or 0.0)
    if direct_citation:
        base = 1.0
    return max(0.0, min(1.0, base))


def s_data(i: PaperFeatures, j: PaperFeatures) -> float | None:
    if not (i.datasets or j.datasets):
        return None
    return jaccard(i.datasets, j.datasets)


# ---------------------------------------------------------------------------
# Temporal factor
# ---------------------------------------------------------------------------
def temporal_factor(
    i: PaperFeatures,
    j: PaperFeatures,
    weights: SimilarityWeights,
    now: datetime | None = None,
) -> float:
    """Recency boost for edges touching recently ingested papers.

    Old foundational links are NOT decayed (they still matter); we only multiply
    up edges that touch a paper ingested within ``recency_days`` so "what's new"
    views surface them.
    """
    now = now or datetime.now(UTC)
    boost = 1.0
    for feat in (i, j):
        if feat.ingested_at is None:
            continue
        ingested = feat.ingested_at
        if ingested.tzinfo is None:
            ingested = ingested.replace(tzinfo=UTC)
        age_days = (now - ingested).total_seconds() / 86400.0
        if 0 <= age_days <= weights.recency_days:
            boost = 1.0 + weights.recency_boost
            break
    return boost


# ---------------------------------------------------------------------------
# The composite
# ---------------------------------------------------------------------------
def composite_weight(
    i: PaperFeatures,
    j: PaperFeatures,
    weights: SimilarityWeights,
    calibrator: CosineCalibrator | None = None,
    *,
    co_citation: float | None = None,
    direct_citation: bool = False,
    now: datetime | None = None,
) -> WeightResult:
    """Compute the full composite edge weight with auditable anatomy.

    Missing components are dropped and the remaining coefficients renormalized so
    a paper lacking, say, citation data is not penalized into oblivion. The set of
    components that actually contributed is returned for the audit log.
    """
    calibrator = calibrator or CosineCalibrator()

    raw_components: dict[str, tuple[float | None, float]] = {
        "sem": (s_sem(i, j, calibrator), weights.alpha),
        "meth": (s_meth(i, j, weights.meth_section_weight), weights.beta),
        "cit": (
            s_cit(i, j, co_citation=co_citation, direct_citation=direct_citation),
            weights.gamma,
        ),
        "data": (s_data(i, j), weights.delta),
    }

    available = {k: (v, w) for k, (v, w) in raw_components.items() if v is not None}
    components = {k: v for k, (v, _w) in available.items()}

    coeff_sum = sum(w for _v, w in available.values())
    if coeff_sum <= 0:
        # Nothing to go on: no edge.
        return WeightResult(
            weight=0.0,
            raw_score=0.0,
            temporal_factor=1.0,
            components=components,
            available=sorted(available),
            coefficients={},
        )

    eff_coeffs = {k: w / coeff_sum for k, (_v, w) in available.items()}
    raw_score = sum(eff_coeffs[k] * components[k] for k in available)

    squashed = sigmoid(raw_score, weights.sigmoid_gain, weights.sigmoid_midpoint)
    t = temporal_factor(i, j, weights, now=now)
    weight = min(1.0, squashed * t)

    return WeightResult(
        weight=weight,
        raw_score=raw_score,
        temporal_factor=t,
        components=components,
        available=sorted(available),
        coefficients=eff_coeffs,
    )
