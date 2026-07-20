from __future__ import annotations

import json

import pytest
from lattice.config import PostgresSettings, Settings, SimilarityWeights, get_settings
from lattice.core.cost import CostTracker, Usage, estimate_cost, price_for
from lattice.core.errors import CostCapExceeded
from lattice.core.hashing import (
    content_hash,
    normalize_arxiv,
    normalize_doi,
    normalize_text,
    stable_id,
)
from lattice.core.llm import LLMResponse


# --------------------------------------------------------------- LLM JSON salvage
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',  # clean
        '```json\n{"a": 1}\n```',  # fenced
        '```\n{"a": 1}\n```',  # fenced without language tag
        'Here is the JSON you asked for:\n```json\n{"a": 1}\n```\nHope this helps!',  # prose + fence
        'Sure! The result is {"a": 1} as requested.',  # bare prose wrapping
    ],
)
def test_llm_response_json_salvages_common_wrappings(text: str) -> None:
    assert LLMResponse(text=text).json() == {"a": 1}


def test_llm_response_json_nested_braces_in_prose() -> None:
    # first-{ to last-} must span the whole object, not stop at an inner brace.
    out = LLMResponse(text='prefix {"a": {"b": [1, 2]}} suffix').json()
    assert out == {"a": {"b": [1, 2]}}


def test_llm_response_json_top_level_array_passes_through() -> None:
    assert LLMResponse(text="[1, 2, 3]").json() == [1, 2, 3]


def test_llm_response_json_garbage_still_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        LLMResponse(text="I could not produce any JSON, sorry.").json()
    with pytest.raises(json.JSONDecodeError):
        LLMResponse(text="broken {not json} here").json()


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


def test_production_settings_require_auth_and_restrict_cors() -> None:
    with pytest.raises(ValueError, match="AUTH_TOKEN"):
        Settings(environment="prod", auth_token=None)
    with pytest.raises(ValueError, match="wildcard CORS"):
        Settings(environment="prod", auth_token="secret", cors_origins=["*"])
    settings = Settings(
        environment="prod",
        auth_token="secret",
        cors_origins=["https://lattice.example"],
    )
    assert settings.cors_origins == ["https://lattice.example"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_workspaces", 0),
        ("rate_limit_per_min", -1),
        ("max_upload_mb", 0),
        ("ingest_max_attempts", 0),
        ("readiness_timeout_s", 0),
    ],
)
def test_settings_reject_unsafe_runtime_limits(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        Settings(**{field: value})


def test_similarity_settings_require_valid_probabilities_and_a_nonzero_weight() -> None:
    with pytest.raises(ValueError):
        SimilarityWeights(tau=1.1)
    with pytest.raises(ValueError, match="at least one similarity weight"):
        SimilarityWeights(alpha=0, beta=0, gamma=0, delta=0)


def test_postgres_settings_require_ordered_pool_limits() -> None:
    with pytest.raises(ValueError, match="pool_max"):
        PostgresSettings(pool_min=5, pool_max=4)


@pytest.mark.parametrize("workspace", ["", "../escape", "white space", "x" * 65])
def test_settings_reject_invalid_workspace_ids(workspace: str) -> None:
    with pytest.raises(ValueError, match="workspace id"):
        Settings(workspace_id=workspace)
