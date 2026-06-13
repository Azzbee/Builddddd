from __future__ import annotations

import pytest
from lattice.embeddings.base import HashingEmbedder
from lattice.graph.store import FakeGraphStore
from lattice.mcp_server import MCP_TOOL_NAMES, call_mcp_tool, mcp_tool_specs
from lattice.rag.models import ChunkHit
from lattice.rag.tools import Toolbox


class FakeVectorStore:
    def __init__(self, hits: list[ChunkHit]) -> None:
        self.hits = hits

    async def hybrid_search(self, query_text, query_embedding, k, filters=None):
        return self.hits[:k]


class FakeCardStore:
    def __init__(self, cards: dict[str, dict]) -> None:
        self.cards = cards

    async def get_card(self, paper_id: str):
        return self.cards.get(paper_id)


def _toolbox() -> Toolbox:
    hit = ChunkHit("c1", "p1", "Copper paper", "Results", "RMSE 0.12", 0.9, "Table 3")
    return Toolbox(
        graph=FakeGraphStore(),
        vectors=FakeVectorStore([hit]),
        cards=FakeCardStore({"p1": {"paper_id": "p1", "title": "T"}}),
        embedder=HashingEmbedder(dim=16),
    )


def test_mcp_tool_specs_are_read_only_subset() -> None:
    names = {t["name"] for t in mcp_tool_specs()}
    assert names == set(MCP_TOOL_NAMES)
    # No write/mutating tools exposed.
    assert "run_cypher" not in names


async def test_call_mcp_tool_dispatches() -> None:
    tb = _toolbox()
    out = await call_mcp_tool(tb, "search_chunks", {"query": "rmse"})
    assert out["chunks"][0]["paper_id"] == "p1"
    card = await call_mcp_tool(tb, "get_paper_card", {"paper_id": "p1"})
    assert card["card"]["title"] == "T"


async def test_call_mcp_tool_rejects_unexposed() -> None:
    with pytest.raises(ValueError):
        await call_mcp_tool(_toolbox(), "run_cypher", {"query": "MATCH (n) RETURN n"})
