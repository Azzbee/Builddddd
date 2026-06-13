from __future__ import annotations

from lattice.core.metrics import MetricsRegistry


def test_counter_increments_and_renders() -> None:
    reg = MetricsRegistry()
    reg.inc("reqs", {"method": "GET", "status": "200"}, help="requests")
    reg.inc("reqs", {"method": "GET", "status": "200"})
    reg.inc("reqs", {"method": "POST", "status": "201"})
    out = reg.render()
    assert "# TYPE reqs counter" in out
    assert 'reqs{method="GET",status="200"} 2.0' in out
    assert 'reqs{method="POST",status="201"} 1.0' in out


def test_histogram_buckets_sum_count() -> None:
    reg = MetricsRegistry()
    for v in (0.001, 0.2, 3.0):
        reg.observe("latency", v, {"path": "/x"})
    out = reg.render()
    assert "# TYPE latency histogram" in out
    assert 'latency_count{path="/x"} 3' in out
    assert 'latency_sum{path="/x"} 3.201' in out
    assert 'latency_bucket{path="/x",le="+Inf"} 3' in out
    # Cumulative bucket at le=0.25 should include the 0.001 and 0.2 observations.
    assert 'latency_bucket{path="/x",le="0.25"} 2' in out


def test_label_escaping() -> None:
    reg = MetricsRegistry()
    reg.inc("c", {"k": 'a"b\\c'})
    assert '\\"' in reg.render()


def test_reset() -> None:
    reg = MetricsRegistry()
    reg.inc("c")
    reg.reset()
    assert reg.render().strip() == ""
