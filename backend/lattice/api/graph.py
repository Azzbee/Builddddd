from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(require_auth)])


@router.get("")
async def graph(
    min_weight: float = Query(0.0, ge=0.0, le=1.0),
    year_from: int | None = None,
    year_to: int | None = None,
    as_of_year: int | None = Query(None, description="Reconstruct the graph as of this publication year"),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Nodes + edges for the explorer, filtered by weight and year.

    ``as_of_year`` time-travels: the graph is rebuilt with only papers published up
    to that year (communities/centrality recomputed on the subgraph).
    """
    snapshot = await c.ingestion.graph_snapshot(as_of_year=as_of_year)
    nodes = snapshot.nodes
    edges = snapshot.edges

    if year_from is not None or year_to is not None:
        def in_range(year: int | None) -> bool:
            if year is None:
                return False
            if year_from is not None and year < year_from:
                return False
            return not (year_to is not None and year > year_to)

        nodes = [n for n in nodes if in_range(n.year)]
        keep = {n.id for n in nodes}
        edges = [e for e in edges if e.source in keep and e.target in keep]

    edges = [e for e in edges if e.weight >= min_weight]
    return {"nodes": [n.to_json() for n in nodes], "edges": [e.to_json() for e in edges]}


@router.get("/stats")
async def stats(c: Container = Depends(get_container)) -> dict[str, object]:
    snapshot = await c.ingestion.graph_snapshot()
    communities = {n.community for n in snapshot.nodes}
    return {
        "papers": len(snapshot.nodes),
        "edges": len(snapshot.edges),
        "communities": len(communities),
    }


@router.get("/timeline")
async def timeline(c: Container = Depends(get_container)) -> dict[str, object]:
    """Publication-year bounds + cumulative growth, for the time-travel slider."""
    return await c.ingestion.graph_timeline()


@router.get("/delta")
async def delta(
    since_year: int = Query(..., description="Lower bound (exclusive) on the year axis"),
    until_year: int | None = Query(None, description="Upper bound; defaults to now"),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    """Papers and edges that entered the field in (since_year, until_year]."""
    return await c.ingestion.graph_delta(since_year, until_year)
