from __future__ import annotations

from lattice.graph.lineage import LineageNode, build_lineage


def _nodes() -> list[LineageNode]:
    return [
        LineageNode("p1", "Seminal LSTM", 2016, {"lstm"}),
        LineageNode("p2", "LSTM + attention", 2019, {"lstm", "attention"}),
        LineageNode("p3", "Transformer forecasting", 2021, {"transformer"}),
        LineageNode("p4", "LSTM survey", 2022, {"lstm"}),
    ]


def test_build_lineage_filters_by_method_and_orders_by_year() -> None:
    lin = build_lineage(_nodes(), cite_edges=[("p2", "p1"), ("p4", "p1")], method="lstm")
    ids = [n.paper_id for n in lin.nodes]
    assert ids == ["p1", "p2", "p4"]  # transformer paper excluded; sorted by year
    # Citation (p2 cites p1) reoriented older->newer.
    assert any(e.source == "p1" and e.target == "p2" for e in lin.edges)


def test_lineage_excludes_cross_method_edges() -> None:
    lin = build_lineage(_nodes(), cite_edges=[("p3", "p1")], method="lstm")
    # p3 is a transformer paper, not in the lstm family -> edge dropped.
    assert all("p3" not in (e.source, e.target) for e in lin.edges)


def test_lineage_timeline_grouping() -> None:
    lin = build_lineage(_nodes(), cite_edges=[], method="lstm").to_json()
    timeline = lin["timeline"]
    assert "2016" in timeline and "2019" in timeline and "2022" in timeline
    assert timeline["2016"] == ["p1"]


def test_lineage_extends_edges() -> None:
    lin = build_lineage(
        _nodes(), cite_edges=[], method="lstm", extends_edges=[("p1", "p2")]
    )
    assert any(e.kind == "extends" and e.source == "p1" and e.target == "p2" for e in lin.edges)
