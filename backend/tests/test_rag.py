from __future__ import annotations

import json
from typing import Any

import pytest
from lattice.config import RagSettings
from lattice.core.llm import LLMMessage, LLMResponse
from lattice.embeddings.base import HashingEmbedder
from lattice.graph.store import FakeGraphStore
from lattice.rag.agent import RagAgent, citations_to_markdown
from lattice.rag.models import ChunkHit, QueryClass
from lattice.rag.router import classify
from lattice.rag.synthesis import Provenance, build_citations
from lattice.rag.tools import Toolbox


# --------------------------------------------------------------------------- router
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What dataset did paper X use?", QueryClass.FACTUAL),
        ("Compare the eval protocols of these 3 papers", QueryClass.COMPARATIVE),
        ("Which papers build on Y's method?", QueryClass.RELATIONAL),
        ("What are the main schools of thought in this field?", QueryClass.GLOBAL),
    ],
)
def test_router_classification(query: str, expected: QueryClass) -> None:
    assert classify(query) == expected


# --------------------------------------------------------------------------- synthesis
def test_build_citations_drops_ungrounded() -> None:
    prov = Provenance()
    prov.add_paper("p1", "Paper One", "Table 1")
    claimed = [
        {"paper_id": "p1", "evidence_location": "Table 1"},
        {"paper_id": "ghost", "evidence_location": "made up"},  # never retrieved
        {"paper_id": "p1"},  # duplicate
    ]
    cites = build_citations(claimed, prov)
    assert len(cites) == 1
    assert cites[0].paper_id == "p1" and cites[0].marker == 1


# --------------------------------------------------------------------------- tools
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


def _toolbox(graph=None, hits=None, cards=None) -> Toolbox:
    return Toolbox(
        graph=graph or FakeGraphStore(),
        vectors=FakeVectorStore(hits or []),
        cards=FakeCardStore(cards or {}),
        embedder=HashingEmbedder(dim=32),
        workspace_id="ws1",
    )


async def test_tool_search_chunks_returns_provenance() -> None:
    hit = ChunkHit("c1", "p1", "Copper paper", "Results", "RMSE 0.12", 0.9, "Table 3")
    tb = _toolbox(hits=[hit])
    out = await tb.call("search_chunks", {"query": "rmse"})
    assert out["chunks"][0]["paper_id"] == "p1"
    assert out["chunks"][0]["evidence_location"] == "Table 3"


async def test_tool_get_paper_card() -> None:
    tb = _toolbox(cards={"p1": {"paper_id": "p1", "title": "T"}})
    assert (await tb.call("get_paper_card", {"paper_id": "p1"}))["card"]["title"] == "T"
    assert (await tb.call("get_paper_card", {"paper_id": "x"}))["error"] == "not_found"


async def test_tool_graph_neighbors_builds_typed_traversal() -> None:
    graph = FakeGraphStore({"RETURN DISTINCT n.id": [{"id": "p2", "title": "N", "weight": 0.8}]})
    tb = _toolbox(graph=graph)
    out = await tb.call("graph_neighbors", {"paper_id": "p1", "edge_types": ["RELATED_TO"], "hops": 2})
    assert out["neighbors"][0]["id"] == "p2"
    q = graph.calls[-1][0]
    assert "RELATED_TO*1..2" in q


async def test_tool_find_papers_filters() -> None:
    graph = FakeGraphStore({"RETURN DISTINCT p.id": [{"id": "p1", "title": "T", "year": 2024}]})
    tb = _toolbox(graph=graph)
    out = await tb.call("find_papers", {"method": "LSTM", "year_from": 2020})
    assert out["papers"][0]["id"] == "p1"
    q, params = graph.calls[-1]
    assert "USES_METHOD" in q and params["method"] == "LSTM"


async def test_tool_run_cypher_rejects_writes() -> None:
    tb = _toolbox()
    out = await tb.call("run_cypher", {"query": "MATCH (p) DELETE p"})
    assert "error" in out


async def test_tool_run_cypher_allows_reads() -> None:
    graph = FakeGraphStore({"RETURN": [{"n": 1}]})
    tb = _toolbox(graph=graph)
    out = await tb.call("run_cypher", {"query": "MATCH (p:Paper) RETURN count(p) AS n"})
    assert out["rows"] == [{"n": 1}]


async def test_tool_unknown_raises() -> None:
    with pytest.raises(ValueError):
        await _toolbox().call("nope", {})


# --------------------------------------------------------------------------- agent
class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, model, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.calls += 1
        text = self._responses.pop(0)
        return LLMResponse(text=text, input_tokens=500, output_tokens=200, model=model)


async def test_agent_uses_tool_then_answers_with_citation() -> None:
    hit = ChunkHit("c1", "p1", "Copper paper", "Results", "RMSE 0.12 beats ARIMA", 0.9, "Table 3")
    tb = _toolbox(hits=[hit])
    llm = ScriptedLLM(
        [
            json.dumps({"action": "search_chunks", "args": {"query": "rmse copper"}}),
            json.dumps(
                {
                    "answer": "The LSTM model reached RMSE 0.12, beating ARIMA [1].",
                    "citations": [{"paper_id": "p1", "evidence_location": "Table 3"}],
                    "confidence": 0.9,
                }
            ),
        ]
    )
    agent = RagAgent(llm, tb, RagSettings())
    result = await agent.run("What RMSE did the copper LSTM achieve?")
    assert not result.abstained
    assert result.confidence == 0.9
    assert len(result.citations) == 1
    assert result.citations[0].paper_id == "p1"
    assert len(result.tool_calls) == 1 and result.tool_calls[0].name == "search_chunks"
    assert result.input_tokens > 0


async def test_agent_abstains_on_low_confidence() -> None:
    tb = _toolbox(hits=[])
    llm = ScriptedLLM(
        [json.dumps({"answer": "", "citations": [], "confidence": 0.1})]
    )
    agent = RagAgent(llm, tb, RagSettings())
    result = await agent.run("Something not in the corpus?")
    assert result.abstained
    assert "don't have enough evidence" in result.answer


async def test_agent_streams_events() -> None:
    hit = ChunkHit("c1", "p1", "Paper", "Results", "text", 0.9, "Sec 1")
    tb = _toolbox(hits=[hit])
    llm = ScriptedLLM(
        [
            json.dumps({"action": "search_chunks", "args": {"query": "x"}}),
            json.dumps({"answer": "Answer [1].", "citations": [{"paper_id": "p1"}], "confidence": 0.8}),
        ]
    )
    agent = RagAgent(llm, tb, RagSettings())
    types = [e.type async for e in agent.stream("q")]
    assert "status" in types
    assert "tool_call" in types and "tool_result" in types
    assert types[-1] == "final"


async def test_agent_respects_tool_call_ceiling() -> None:
    tb = _toolbox(hits=[ChunkHit("c", "p1", "P", "S", "t", 0.5, "S")])
    # Always returns a tool action, never a final answer.
    action = json.dumps({"action": "search_chunks", "args": {"query": "x"}})
    llm = ScriptedLLM([action] * 10)
    agent = RagAgent(llm, tb, RagSettings(max_tool_calls=3))
    result = await agent.run("q")
    assert result.abstained
    assert len(result.tool_calls) == 3  # ceiling enforced


async def test_agent_tracks_cost() -> None:
    from lattice.core.cost import CostTracker

    tb = _toolbox(hits=[])
    llm = ScriptedLLM([json.dumps({"answer": "x", "citations": [], "confidence": 0.0})])
    tracker = CostTracker(per_job_cap=10, daily_cap=100)
    agent = RagAgent(llm, tb, RagSettings(), cost=tracker)
    result = await agent.run("q")
    assert tracker.job_usage.cost_usd > 0
    assert result.cost_usd > 0


def test_citations_to_markdown() -> None:
    prov = Provenance()
    prov.add_paper("p1", "Title One", "Table 1")
    cites = build_citations([{"paper_id": "p1"}], prov)
    md = citations_to_markdown(cites)
    assert "[1] Title One" in md


def _unused(_: Any) -> None:
    pass
