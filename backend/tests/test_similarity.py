from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from lattice.config import SimilarityWeights
from lattice.graph.similarity import (
    CosineCalibrator,
    PaperFeatures,
    composite_weight,
    cosine,
    jaccard,
    s_cit,
    s_data,
    s_meth,
    s_sem,
    sigmoid,
    temporal_factor,
)


def test_cosine_basic_and_edge_cases() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, c) == pytest.approx(0.0)
    assert cosine(a, None) is None
    assert cosine(a, np.zeros(2)) is None


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0


def test_sigmoid_monotonic_centered() -> None:
    assert sigmoid(0.5, gain=4.0, midpoint=0.5) == pytest.approx(0.5)
    assert sigmoid(1.0, 4.0, 0.5) > sigmoid(0.0, 4.0, 0.5)


def test_calibrator_rescales_against_corpus() -> None:
    # Corpus cosines cluster high (SPECTER optimism); calibrator spreads them.
    cosines = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    cal = CosineCalibrator.fit(cosines)
    assert cal.fitted
    assert cal.transform(cal.lo) == pytest.approx(0.0)
    assert cal.transform(cal.hi) == pytest.approx(1.0)
    assert cal.transform(0.0) == 0.0  # clamps below
    assert cal.transform(2.0) == 1.0  # clamps above


def test_calibrator_unfit_is_identity_clamp() -> None:
    cal = CosineCalibrator()
    assert cal.transform(0.5) == pytest.approx(0.5)
    assert CosineCalibrator.fit([0.5]).fitted is False


def test_s_meth_blends_section_and_tags() -> None:
    v = np.array([1.0, 0.0])
    i = PaperFeatures("i", methodology_embedding=v, methods={"lstm"})
    j = PaperFeatures("j", methodology_embedding=v, methods={"lstm", "var"})
    # section cosine = 1.0, tag jaccard = 1/2, weight 0.6 -> 0.6*1 + 0.4*0.5 = 0.8
    assert s_meth(i, j, section_weight=0.6) == pytest.approx(0.8)


def test_s_meth_falls_back_when_one_signal_missing() -> None:
    i = PaperFeatures("i", methods={"lstm"})
    j = PaperFeatures("j", methods={"lstm"})
    assert s_meth(i, j, 0.6) == pytest.approx(1.0)  # tags only
    assert s_meth(PaperFeatures("i"), PaperFeatures("j"), 0.6) is None  # nothing


def test_s_cit_direct_citation_saturates() -> None:
    i = PaperFeatures("i", references={"r1"})
    j = PaperFeatures("j", references={"r2"})
    assert s_cit(i, j) == 0.0  # no overlap
    assert s_cit(i, j, direct_citation=True) == 1.0
    coupled = s_cit(
        PaperFeatures("i", references={"r1", "r2"}),
        PaperFeatures("j", references={"r2", "r3"}),
    )
    assert coupled == pytest.approx(1 / 3)
    assert s_cit(i, j, co_citation=0.8) == pytest.approx(0.8)


def test_s_cit_none_when_no_signal() -> None:
    assert s_cit(PaperFeatures("i"), PaperFeatures("j")) is None


def test_s_data_jaccard() -> None:
    i = PaperFeatures("i", datasets={"comex", "lme"})
    j = PaperFeatures("j", datasets={"lme"})
    assert s_data(i, j) == pytest.approx(0.5)
    assert s_data(PaperFeatures("i"), PaperFeatures("j")) is None


def test_s_sem_uses_calibration() -> None:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.7, 0.7])
    cal = CosineCalibrator(lo=0.0, hi=1.0, fitted=True)
    i = PaperFeatures("i", specter=v1)
    j = PaperFeatures("j", specter=v2)
    got = s_sem(i, j, cal)
    assert got is not None and 0.0 <= got <= 1.0
    assert s_sem(PaperFeatures("i"), PaperFeatures("j"), cal) is None


def test_temporal_factor_boosts_recent_only() -> None:
    w = SimilarityWeights()
    now = datetime(2026, 6, 12, tzinfo=UTC)
    recent = PaperFeatures("r", ingested_at=now - timedelta(days=10))
    old = PaperFeatures("o", ingested_at=now - timedelta(days=400))
    assert temporal_factor(recent, old, w, now=now) == pytest.approx(1.0 + w.recency_boost)
    assert temporal_factor(old, old, w, now=now) == pytest.approx(1.0)


def test_composite_weight_full_anatomy() -> None:
    w = SimilarityWeights()
    v = np.array([1.0, 0.0])
    i = PaperFeatures(
        "i", specter=v, methodology_embedding=v, methods={"lstm"}, datasets={"lme"},
        references={"r1", "r2"},
    )
    j = PaperFeatures(
        "j", specter=v, methodology_embedding=v, methods={"lstm"}, datasets={"lme"},
        references={"r1", "r3"},
    )
    res = composite_weight(i, j, w, CosineCalibrator())
    assert set(res.available) == {"sem", "meth", "cit", "data"}
    assert 0.0 < res.weight <= 1.0
    assert sum(res.coefficients.values()) == pytest.approx(1.0)
    # Identical strong signals should produce a high weight.
    assert res.weight > 0.7
    js = res.to_json()
    assert "components" in js and "coefficients" in js


def test_composite_renormalizes_when_components_missing() -> None:
    w = SimilarityWeights()
    v = np.array([1.0, 0.0])
    # Only semantic available: coefficient should renormalize to 1.0.
    i = PaperFeatures("i", specter=v)
    j = PaperFeatures("j", specter=v)
    res = composite_weight(i, j, w, CosineCalibrator())
    assert res.available == ["sem"]
    assert res.coefficients == {"sem": pytest.approx(1.0)}
    assert res.weight > 0.5  # identical vectors, calibrator identity


def test_composite_zero_when_nothing_available() -> None:
    w = SimilarityWeights()
    res = composite_weight(PaperFeatures("i"), PaperFeatures("j"), w)
    assert res.weight == 0.0
    assert res.available == []


def test_unrelated_papers_below_threshold() -> None:
    w = SimilarityWeights()
    cal = CosineCalibrator(lo=0.0, hi=1.0, fitted=True)
    i = PaperFeatures("i", specter=np.array([1.0, 0.0]), methods={"lstm"}, datasets={"a"})
    j = PaperFeatures("j", specter=np.array([0.0, 1.0]), methods={"garch"}, datasets={"b"})
    res = composite_weight(i, j, w, cal)
    # Orthogonal semantics + disjoint methods/data -> should fall under tau.
    assert res.weight < w.tau
