"""LLM cost accounting and spend caps.

Token usage is converted to USD with a static price table (overridable from
config in production). The :class:`CostTracker` enforces per-job and daily caps;
when a cap is hit it raises :class:`CostCapExceeded` so jobs PAUSE rather than
fail, per the PRD's cost-ceiling requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .errors import CostCapExceeded

# USD per 1M tokens (input, output). Approximate June 2026 list prices; the point
# is relative accounting and cap enforcement, not invoicing precision.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "mistral/mistral-large-latest": (2.0, 6.0),
    "bge-m3": (0.0, 0.0),
    "specter2": (0.0, 0.0),
}

_DEFAULT_PRICE = (3.0, 15.0)


def price_for(model: str) -> tuple[float, float]:
    """Return (input, output) USD-per-1M-token price for a model."""
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    # Fall back to a family match (e.g. "mistral/...").
    for key, price in PRICE_TABLE.items():
        if model.startswith(key.split("/")[0]):
            return price
    return _DEFAULT_PRICE


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a single completion."""
    in_price, out_price = price_for(model)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


@dataclass
class Usage:
    """Accumulated token usage and cost for one logical unit (job or query)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        return cost


@dataclass
class CostTracker:
    """Enforces per-job and daily spend caps.

    ``daily_spend`` is process-local; in production it is hydrated from Postgres
    at startup so caps survive restarts. The logic is identical either way.
    """

    per_job_cap: float
    daily_cap: float
    job_usage: Usage = field(default_factory=Usage)
    daily_spend: float = 0.0
    _day: date = field(default_factory=date.today)

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self.daily_spend = 0.0

    def check(self) -> None:
        """Raise if either cap is already exceeded. Call before a paid request."""
        self._roll_day()
        if self.job_usage.cost_usd >= self.per_job_cap:
            raise CostCapExceeded(
                f"per-job cap ${self.per_job_cap:.2f} reached "
                f"(spent ${self.job_usage.cost_usd:.4f})"
            )
        if self.daily_spend >= self.daily_cap:
            raise CostCapExceeded(
                f"daily cap ${self.daily_cap:.2f} reached (spent ${self.daily_spend:.4f})"
            )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record usage after a request and return its cost."""
        self._roll_day()
        cost = self.job_usage.add(model, input_tokens, output_tokens)
        self.daily_spend += cost
        return cost
