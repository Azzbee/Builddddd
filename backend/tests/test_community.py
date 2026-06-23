"""Tests for greedy-modularity community detection."""

from __future__ import annotations

from lattice.graph.community import greedy_modularity_communities


def test_two_clusters_with_a_bridge_are_separated() -> None:
    # Two triangles (a,b,c) and (d,e,f) joined by a single weak bridge c-d.
    # Connected components would call this ONE community; modularity finds TWO.
    edges = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0),
        ("d", "e", 1.0), ("e", "f", 1.0), ("d", "f", 1.0),
        ("c", "d", 0.1),  # weak bridge
    ]
    comm = greedy_modularity_communities(["a", "b", "c", "d", "e", "f"], edges)
    assert comm["a"] == comm["b"] == comm["c"]
    assert comm["d"] == comm["e"] == comm["f"]
    assert comm["a"] != comm["d"]
    assert len(set(comm.values())) == 2


def test_isolated_nodes_each_own_community() -> None:
    comm = greedy_modularity_communities(["x", "y", "z"], [])
    assert len(set(comm.values())) == 3


def test_empty_graph() -> None:
    assert greedy_modularity_communities([], []) == {}


def test_deterministic_labels() -> None:
    edges = [("a", "b", 1.0), ("c", "d", 1.0)]
    a = greedy_modularity_communities(["a", "b", "c", "d"], edges)
    b = greedy_modularity_communities(["d", "c", "b", "a"], edges)
    assert a == b  # stable regardless of input order


def test_single_clique_is_one_community() -> None:
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0)]
    comm = greedy_modularity_communities(["a", "b", "c"], edges)
    assert len(set(comm.values())) == 1


def test_ignores_self_loops_and_unknown_nodes() -> None:
    edges = [("a", "a", 5.0), ("a", "b", 1.0), ("b", "ghost", 9.0)]
    comm = greedy_modularity_communities(["a", "b"], edges)
    assert comm["a"] == comm["b"]
