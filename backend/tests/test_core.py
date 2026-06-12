from __future__ import annotations

import pytest
from lattice.config import Settings, get_settings
from lattice.core.cost import CostTracker, Usage, estimate_cost, price_for
from lattice.core.errors import CostCapExceeded
from lattice.core.hashing import (
    content_hash,
    normalize_arxiv,
    normalize_doi,
    normalize_text,
    stable_id,
)


def test_content_hash_is_stable_and_distinct() -> None:
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")
    assert len(content_hash(b"")) == 64


def test_stable_id_deterministic_and_order_sensitive() -> None:
    assert stable_id("method", "LSTM") == stable_id("method", "lstm ")
    assert stable_id("a", "b") != stable_id("b", "a")


def test_normalize_text_strips_accents_punctuation_case() -> None:
    assert normalize_text("  Données  d'Énergie!! ") == "donnees d energie"
    assert normalize_text("LSTM-VAR (hybrid)") == "lstm var hybrid"
    assert normalize_text("") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1/ABC", "10.1/abc"),
        ("doi:10.2/xyz", "10.2/xyz"),
        ("10.3/Q", "10.3/q"),
        (None, None),
    ],
)
def test_normalize_doi(raw: str | None, expected: str | None) -> None:
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("arXiv:2401.01234v2", "2401.01234"),
        ("https://arxiv.org/abs/2401.01234", "2401.01234"),
        ("https://arxiv.org/pdf/2401.01234v3.pdf", "2401.01234"),
        (None, None),
    ],
)
def test_normalize_arxiv(raw: str | None, expected: str | None) -> None:
    assert normalize_arxiv(raw) == expected


def test_price_and_cost_estimation() -> None:
    assert price_for("claude-haiku-4-5-20251001") == (0.80, 4.0)
    # Unknown model falls back to family or default.
    assert price_for("mistral/whatever") == (2.0, 6.0)
    cost = estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
    assert cost == pytest.approx(4.80)


def test_usage_accumulates_by_model() -> None:
    u = Usage()
    u.add("claude-haiku-4-5-20251001", 1000, 500)
    u.add("claude-sonnet-4-6", 1000, 500)
    assert u.calls == 2
    assert set(u.by_model) == {"claude-haiku-4-5-20251001", "claude-sonnet-4-6"}
    assert u.cost_usd > 0


def test_cost_tracker_enforces_per_job_cap() -> None:
    t = CostTracker(per_job_cap=0.01, daily_cap=100.0)
    t.check()  # fine at zero
    t.record("claude-sonnet-4-6", 10_000, 10_000)  # well over $0.01
    with pytest.raises(CostCapExceeded):
        t.check()


def test_cost_tracker_enforces_daily_cap() -> None:
    t = CostTracker(per_job_cap=100.0, daily_cap=0.01)
    t.record("claude-sonnet-4-6", 10_000, 10_000)
    with pytest.raises(CostCapExceeded):
        t.check()


def test_settings_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATTICE_SIMILARITY__ALPHA", "0.5")
    monkeypatch.setenv("LATTICE_WORKSPACE_ID", "proj-x")
    get_settings.cache_clear()
    s = Settings()
    assert s.similarity.alpha == 0.5
    assert s.workspace_id == "proj-x"


def test_similarity_weight_negative_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATTICE_SIMILARITY__ALPHA", "-1")
    with pytest.raises(ValueError):
        Settings()
