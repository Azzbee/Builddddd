"""Graph analytics via Neo4j GDS: PageRank (centrality), Louvain (communities),
betweenness (bridges). Results are written back onto Paper nodes so the explorer
can size nodes by PageRank and color by community.

The Cypher is built here; execution goes through the GraphStore so the read paths
are testable with canned results and the algorithm wiring is assertable.
"""

from __future__ import annotations

from typing import Any

from lattice.graph.store import GraphStore

_GRAPH_NAME = "lattice_related"


async def _drop_projection(store: GraphStore) -> None:
    await store.execute(
        "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName",
        {"name": _GRAPH_NAME},
    )


async def _project(store: GraphStore, workspace_id: str) -> None:
    """Project the valid RELATED_TO subgraph for one workspace, weighted by edge weight."""
    await _drop_projection(store)
    await store.execute(
        """
        MATCH (src:Paper {workspace_id: $ws})-[r:RELATED_TO]->(tgt:Paper {workspace_id: $ws})
        WHERE r.invalid_at IS NULL
        WITH gds.graph.project($name, src, tgt,
            {relationshipProperties: r {.weight}}) AS g
        RETURN g.graphName AS name
        """,
        {"ws": workspace_id, "name": _GRAPH_NAME},
    )


async def recompute_analytics(store: GraphStore, workspace_id: str = "default") -> dict[str, Any]:
    """Run PageRank, Louvain, and betweenness, writing results back to nodes.

    Returns a small summary (node/community counts) for logging and the digest.
    """
    await _project(store, workspace_id)
    await store.execute(
        """
        CALL gds.pageRank.write($name, {writeProperty: 'pagerank', relationshipWeightProperty: 'weight'})
        YIELD nodePropertiesWritten RETURN nodePropertiesWritten
        """,
        {"name": _GRAPH_NAME},
    )
    await store.execute(
        """
        CALL gds.louvain.write($name, {writeProperty: 'community', relationshipWeightProperty: 'weight'})
        YIELD communityCount RETURN communityCount
        """,
        {"name": _GRAPH_NAME},
    )
    await store.execute(
        """
        CALL gds.betweenness.write($name, {writeProperty: 'betweenness'})
        YIELD nodePropertiesWritten RETURN nodePropertiesWritten
        """,
        {"name": _GRAPH_NAME},
    )
    await _drop_projection(store)

    rows = await store.execute(
        """
        MATCH (p:Paper {workspace_id: $ws})
        RETURN count(p) AS papers, count(DISTINCT p.community) AS communities
        """,
        {"ws": workspace_id},
    )
    return rows[0] if rows else {"papers": 0, "communities": 0}


async def top_papers_by_pagerank(
    store: GraphStore, workspace_id: str = "default", limit: int = 10
) -> list[dict[str, Any]]:
    return await store.execute(
        """
        MATCH (p:Paper {workspace_id: $ws})
        WHERE p.pagerank IS NOT NULL
        RETURN p.id AS id, p.title AS title, p.pagerank AS pagerank, p.community AS community
        ORDER BY p.pagerank DESC
        LIMIT $limit
        """,
        {"ws": workspace_id, "limit": limit},
    )


async def community_members(
    store: GraphStore, community: int, workspace_id: str = "default"
) -> list[dict[str, Any]]:
    return await store.execute(
        """
        MATCH (p:Paper {workspace_id: $ws, community: $community})
        RETURN p.id AS id, p.title AS title, p.pagerank AS pagerank
        ORDER BY p.pagerank DESC
        """,
        {"ws": workspace_id, "community": community},
    )
