"""MCP server: expose the Lattice graph as tools for Claude Desktop/Code.

Reuses the exact RAG ``Toolbox``, so the model that writes alongside you can query
your knowledge graph with the same tools the chat agent uses. The subset exposed is
read-only and side-effect-free.

Run it with the ``mcp`` extra installed:

    uv run python -m lattice.mcp_server   # stdio transport
"""

from __future__ import annotations

from typing import Any

from lattice.rag.tools import TOOL_SCHEMAS, Toolbox

#: The read-only subset exposed over MCP (PRD extra #8).
MCP_TOOL_NAMES = ("search_chunks", "find_papers", "graph_neighbors", "get_paper_card")


def mcp_tool_specs() -> list[dict[str, Any]]:
    """Return the schemas for the MCP-exposed tools (pure, testable)."""
    return [t for t in TOOL_SCHEMAS if t["name"] in MCP_TOOL_NAMES]


async def call_mcp_tool(toolbox: Toolbox, name: str, args: dict[str, Any]) -> Any:
    """Dispatch an MCP tool call to the toolbox, enforcing the allowlist."""
    if name not in MCP_TOOL_NAMES:
        raise ValueError(f"tool {name!r} is not exposed over MCP")
    return await toolbox.call(name, args)


def build_fastmcp(toolbox: Toolbox) -> Any:  # pragma: no cover - requires the mcp package
    """Build a FastMCP server wired to the toolbox. Requires ``mcp``."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("lattice")

    @server.tool()
    async def search_chunks(query: str, k: int = 12) -> Any:
        """Hybrid vector + keyword search over paper text for specific facts."""
        return await call_mcp_tool(toolbox, "search_chunks", {"query": query, "k": k})

    @server.tool()
    async def find_papers(
        method: str | None = None,
        dataset: str | None = None,
        concept: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> Any:
        """Structured filter for papers by method/dataset/concept/year range."""
        return await call_mcp_tool(
            toolbox,
            "find_papers",
            {
                "method": method,
                "dataset": dataset,
                "concept": concept,
                "year_from": year_from,
                "year_to": year_to,
            },
        )

    @server.tool()
    async def graph_neighbors(paper_id: str, min_weight: float = 0.0, hops: int = 1) -> Any:
        """Typed graph traversal from a paper along RELATED_TO edges."""
        return await call_mcp_tool(
            toolbox,
            "graph_neighbors",
            {"paper_id": paper_id, "min_weight": min_weight, "hops": hops},
        )

    @server.tool()
    async def get_paper_card(paper_id: str) -> Any:
        """Fetch the full structured card for a paper by id."""
        return await call_mcp_tool(toolbox, "get_paper_card", {"paper_id": paper_id})

    return server


def _build_toolbox() -> Toolbox:  # pragma: no cover - wiring
    from lattice.api.deps import build_container

    c = build_container()
    return Toolbox(
        graph=c.graph,
        vectors=c.vectors,
        cards=c.cards,
        embedder=c.chunk_embedder,
        workspace_id=c.settings.workspace_id,
    )


def main() -> None:  # pragma: no cover - entrypoint
    server = build_fastmcp(_build_toolbox())
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
