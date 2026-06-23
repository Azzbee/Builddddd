"""Community detection for the in-memory explorer path.

The persistent path reads precomputed Louvain communities from Neo4j GDS. Offline
(demo/dev) we need real communities too - connected-components (union-find)
collapses any connected graph into a single blob, which is useless for a small
corpus. This module implements weighted **greedy modularity** maximization
(agglomerative Clauset-Newman-Moore): start with every node in its own community
and repeatedly merge the pair with the largest positive modularity gain. It is
pure, deterministic (lexicographic tie-breaks), and dependency-free.
"""

from __future__ import annotations

from collections import defaultdict


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _relabel(comm: dict[str, str], nodes: list[str]) -> dict[str, int]:
    """Map raw community roots to dense integer ids, stable in node order."""
    labels: dict[str, int] = {}
    return {n: labels.setdefault(comm[n], len(labels)) for n in nodes}


def greedy_modularity_communities(
    node_ids: list[str], edges: list[tuple[str, str, float]]
) -> dict[str, int]:
    """Partition nodes into communities by greedy modularity maximization.

    ``edges`` are undirected ``(a, b, weight)`` triples. Returns ``node -> community
    id``. Isolated nodes each get their own community. O(merges x links); fine for
    the in-memory corpus scale (the persistent path uses Neo4j GDS instead).
    """
    nodes = sorted(set(node_ids))
    if not nodes:
        return {}

    deg: dict[str, float] = defaultdict(float)
    edge_w: dict[tuple[str, str], float] = defaultdict(float)
    for a, b, wt in edges:
        if a == b or wt <= 0 or a not in node_ids or b not in node_ids:
            continue
        deg[a] += wt
        deg[b] += wt
        edge_w[_pair(a, b)] += wt

    two_m = sum(deg.values())
    comm: dict[str, str] = {n: n for n in nodes}
    if two_m == 0:  # no edges -> every node is its own community
        return _relabel(comm, nodes)

    tot: dict[str, float] = defaultdict(float, {n: deg.get(n, 0.0) for n in nodes})
    links: dict[tuple[str, str], float] = defaultdict(float)
    for (a, b), wt in edge_w.items():
        links[_pair(comm[a], comm[b])] += wt

    while True:
        best_gain = 1e-12
        best: tuple[str, str] | None = None
        # Sorted iteration => deterministic: the lexicographically first pair wins ties.
        for (ca, cb), lw in sorted(links.items()):
            if ca == cb:
                continue
            gain = 2.0 * (lw / two_m - (tot[ca] * tot[cb]) / (two_m * two_m))
            if gain > best_gain + 1e-12:
                best_gain = gain
                best = (ca, cb)
        if best is None:
            break

        ca, cb = best  # merge cb into ca
        for n in nodes:
            if comm[n] == cb:
                comm[n] = ca
        tot[ca] += tot[cb]
        tot[cb] = 0.0
        merged: dict[tuple[str, str], float] = defaultdict(float)
        for (x, y), wt in links.items():
            nx = ca if x == cb else x
            ny = ca if y == cb else y
            merged[_pair(nx, ny)] += wt
        links = merged

    return _relabel(comm, nodes)
