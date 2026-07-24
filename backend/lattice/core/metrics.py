"""A tiny, dependency-free metrics registry that renders Prometheus text format.

Keeping this in-process (no prometheus_client requirement) means metrics are
always available and unit-testable. Counters and histograms cover the API's needs;
``render()`` produces the standard exposition format scraped at ``/metrics``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# Latency buckets in seconds (Prometheus convention).
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

Labels = tuple[tuple[str, str], ...]


def _label_key(labels: dict[str, str] | None) -> Labels:
    return tuple(sorted((labels or {}).items()))


def _fmt_labels(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return "{" + inner + "}"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


@dataclass
class _Counter:
    help: str
    values: dict[Labels, float] = field(default_factory=dict)

    def inc(self, labels: Labels, amount: float) -> None:
        self.values[labels] = self.values.get(labels, 0.0) + amount


@dataclass
class _Histogram:
    help: str
    buckets: tuple[float, ...] = _BUCKETS
    counts: dict[Labels, list[int]] = field(default_factory=dict)
    sums: dict[Labels, float] = field(default_factory=dict)
    totals: dict[Labels, int] = field(default_factory=dict)

    def observe(self, labels: Labels, value: float) -> None:
        bucket_counts = self.counts.setdefault(labels, [0] * (len(self.buckets) + 1))
        placed = False
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                bucket_counts[i] += 1
                placed = True
                break
        if not placed:
            bucket_counts[-1] += 1
        self.sums[labels] = self.sums.get(labels, 0.0) + value
        self.totals[labels] = self.totals.get(labels, 0) + 1


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, _Counter] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str = "") -> _Counter:
        with self._lock:
            return self._counters.setdefault(name, _Counter(help=help))

    def histogram(self, name: str, help: str = "") -> _Histogram:
        with self._lock:
            return self._histograms.setdefault(name, _Histogram(help=help))

    def inc(
        self, name: str, labels: dict[str, str] | None = None, amount: float = 1.0, help: str = ""
    ) -> None:
        self.counter(name, help).inc(_label_key(labels), amount)

    def observe(
        self, name: str, value: float, labels: dict[str, str] | None = None, help: str = ""
    ) -> None:
        self.histogram(name, help).observe(_label_key(labels), value)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    def render(self) -> str:
        lines: list[str] = []
        for name, counter in sorted(self._counters.items()):
            if counter.help:
                lines.append(f"# HELP {name} {counter.help}")
            lines.append(f"# TYPE {name} counter")
            for labels, value in sorted(counter.values.items()):
                lines.append(f"{name}{_fmt_labels(labels)} {value}")
        for name, hist in sorted(self._histograms.items()):
            if hist.help:
                lines.append(f"# HELP {name} {hist.help}")
            lines.append(f"# TYPE {name} histogram")
            for labels, counts in sorted(hist.counts.items()):
                cumulative = 0
                for i, edge in enumerate(hist.buckets):
                    cumulative += counts[i]
                    le = (*labels, ("le", str(edge)))
                    lines.append(f"{name}_bucket{_fmt_labels(le)} {cumulative}")
                cumulative += counts[-1]
                inf = (*labels, ("le", "+Inf"))
                lines.append(f"{name}_bucket{_fmt_labels(inf)} {cumulative}")
                lines.append(f"{name}_sum{_fmt_labels(labels)} {hist.sums.get(labels, 0.0)}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {hist.totals.get(labels, 0)}")
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()
