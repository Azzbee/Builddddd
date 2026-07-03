"""The tool-using RAG agent.

A provider-agnostic ReAct loop: the model returns JSON that is either a tool
action ``{"action": name, "args": {...}}`` or a final answer
``{"answer": ..., "citations": [...], "confidence": 0-1}``. This keeps the agent
fully testable offline with a scripted model and portable across providers (real
provider tool-calling can be swapped in behind the same loop).

Guarantees from the PRD:
* max 8 tool calls per query (configurable)
* "I don't know" is a valid output: low confidence -> abstain, do not synthesize fog
* every answer carries citations validated against retrieved provenance
* token + cost accounting per query
* a tool trace is returned for the "how I got this" panel
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from lattice.config import RagSettings
from lattice.core.cost import CostTracker
from lattice.core.llm import LLMClient, LLMMessage
from lattice.core.logging import get_logger
from lattice.rag.models import AgentEvent, AgentResult, Citation, ToolCall
from lattice.rag.router import classify
from lattice.rag.synthesis import Provenance, build_citations
from lattice.rag.tools import TOOL_SCHEMAS, Toolbox

log = get_logger("rag.agent")

_SYSTEM = """You are Lattice's research assistant. Answer questions about a corpus \
of scientific papers using ONLY the provided tools and their results.

Protocol: respond with a single JSON object and nothing else. Either call a tool:
  {{"action": "<tool_name>", "args": {{...}}}}
or give the final answer:
  {{"answer": "<text with inline [n] citation markers>",
   "citations": [{{"paper_id": "...", "evidence_location": "..."}}],
   "confidence": <0..1>}}

Rules:
- Ground every claim in tool results. Cite with [1], [2] markers that match the
  order of the citations array.
- Never cite a paper you did not retrieve via a tool.
- If the tools do not contain enough evidence to answer, set "confidence" below
  {floor} and say you don't know rather than guessing.
- Be concise and specific. Prefer numbers and named methods/datasets.

Available tools:
{tools}

Query class (hint): {query_class}
"""


class RagAgent:
    def __init__(
        self,
        llm: LLMClient,
        toolbox: Toolbox,
        settings: RagSettings | None = None,
        cost: CostTracker | None = None,
    ) -> None:
        self._llm = llm
        self._tools = toolbox
        self._settings = settings or RagSettings()
        self._cost = cost

    def _system_prompt(self, query_class: str) -> str:
        tool_lines = "\n".join(
            f"- {t['name']}({', '.join(t['parameters'])}): {t['description']}" for t in TOOL_SCHEMAS
        )
        return _SYSTEM.format(
            floor=self._settings.low_confidence_floor,
            tools=tool_lines,
            query_class=query_class,
        )

    async def run(self, query: str) -> AgentResult:
        result: AgentResult | None = None
        async for event in self.stream(query):
            if event.type == "final":
                final = event.data["result"]
                assert isinstance(final, AgentResult)
                result = final
        assert result is not None
        return result

    async def stream(self, query: str) -> AsyncIterator[AgentEvent]:
        qclass = classify(query)
        yield AgentEvent("status", {"message": "classified", "query_class": str(qclass)})

        provenance = Provenance()
        tool_calls: list[ToolCall] = []
        in_tokens = out_tokens = 0
        cost_usd = 0.0

        messages = [
            LLMMessage("system", self._system_prompt(str(qclass))),
            LLMMessage("user", query),
        ]

        final: AgentResult | None = None
        # Iterate enough to exhaust the tool budget and still get a final answer.
        max_iters = self._settings.max_tool_calls * 2 + 2
        for _step in range(max_iters):
            if self._cost is not None:
                self._cost.check()
            resp = await self._llm.complete(
                self._settings.agent_model, messages, json_mode=True, max_tokens=2048
            )
            in_tokens += resp.input_tokens
            out_tokens += resp.output_tokens
            if self._cost is not None:
                cost_usd += self._cost.record(
                    self._settings.agent_model, resp.input_tokens, resp.output_tokens
                )

            try:
                payload = resp.json()
            except json.JSONDecodeError:
                messages.append(LLMMessage("assistant", resp.text))
                messages.append(LLMMessage("user", "Respond with a single valid JSON object only."))
                continue
            if not isinstance(payload, dict):
                # Valid JSON, but a top-level array/string/number/null. `"action" in`
                # and `.get` below assume an object, so re-prompt instead of crashing
                # out of the loop (which would 500 /query and agent.run()).
                messages.append(LLMMessage("assistant", resp.text))
                messages.append(LLMMessage("user", "Respond with a single valid JSON object only."))
                continue

            if "action" in payload:
                if len(tool_calls) >= self._settings.max_tool_calls:
                    # Tool budget exhausted: instruct the model to answer now.
                    messages.append(LLMMessage("assistant", json.dumps(payload)))
                    messages.append(
                        LLMMessage(
                            "user",
                            "Tool-call limit reached. Give your final answer JSON now "
                            "using only what you have already retrieved.",
                        )
                    )
                    continue
                name = str(payload["action"])
                args = payload.get("args") or {}
                yield AgentEvent("tool_call", {"name": name, "args": args})
                call = ToolCall(name=name, args=args)
                try:
                    tool_result = await self._tools.call(name, args)
                    call.result = tool_result
                    _absorb(provenance, tool_result)
                except Exception as exc:
                    call.error = str(exc)
                    tool_result = {"error": str(exc)}
                tool_calls.append(call)
                yield AgentEvent("tool_result", {"name": name, "error": call.error})
                messages.append(LLMMessage("assistant", json.dumps(payload)))
                messages.append(
                    LLMMessage("user", f"Tool {name} result:\n{json.dumps(tool_result)[:8000]}")
                )
                continue

            # Final answer. The model's output shape is untrusted: a syntactically
            # valid JSON object can still carry a non-numeric confidence or a
            # non-list citations field. Coerce defensively so a malformed final
            # answer degrades to a low-confidence abstention instead of raising
            # out of the generator (which would 500 /query and kill the SSE stream).
            answer = str(payload.get("answer", "")).strip()
            confidence = _coerce_confidence(payload.get("confidence"))
            claimed = _coerce_citations(payload.get("citations"))
            citations = build_citations(claimed, provenance)
            abstained = confidence < self._settings.low_confidence_floor or not provenance.papers
            if abstained and not answer:
                answer = "I don't have enough evidence in the corpus to answer that confidently."
            final = AgentResult(
                answer=answer,
                citations=citations,
                tool_calls=tool_calls,
                query_class=qclass,
                confidence=confidence,
                abstained=abstained,
                cost_usd=cost_usd,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            break

        if final is None:
            # Hit the tool-call ceiling without a final answer.
            final = AgentResult(
                answer="I reached the tool-call limit without a confident answer.",
                citations=build_citations([], provenance),
                tool_calls=tool_calls,
                query_class=qclass,
                confidence=0.0,
                abstained=True,
                cost_usd=cost_usd,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )

        log.info(
            "rag.answer",
            query_class=str(qclass),
            tools=len(tool_calls),
            confidence=final.confidence,
            abstained=final.abstained,
            cost_usd=round(final.cost_usd, 6),
        )
        yield AgentEvent("final", {"result": final})


def _coerce_confidence(value: Any) -> float:
    """Best-effort numeric confidence in [0, 1] from untrusted model output.

    A model may emit ``"confidence": "high"``, ``null``, or a stray list; none of
    those should crash the agent. Anything non-numeric maps to 0.0 (abstain).
    """
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf != conf:  # NaN
        return 0.0
    return max(0.0, min(1.0, conf))


def _coerce_citations(value: Any) -> list[dict[str, Any]]:
    """Normalize the model's ``citations`` field to a list of dicts.

    ``build_citations`` iterates and calls ``.get`` on each item, so a string or a
    list of bare ids would raise. Keep only dict entries; drop everything else.
    """
    if not isinstance(value, list):
        return []
    return [c for c in value if isinstance(c, dict)]


def _absorb(provenance: Provenance, result: Any) -> None:
    """Record provenance from a tool result so citations can be validated."""
    if not isinstance(result, dict):
        return
    for chunk in result.get("chunks", []) or []:
        provenance.add_paper(
            chunk.get("paper_id", ""),
            chunk.get("title", ""),
            chunk.get("evidence_location"),
        )
        entry = provenance.papers.get(chunk.get("paper_id", ""))
        if entry and entry.get("snippet") is None:
            entry["snippet"] = (chunk.get("text") or "")[:280]
            entry["page"] = chunk.get("page")
            if chunk.get("section"):
                entry["sections"].add(chunk["section"])
    for key in ("papers", "neighbors"):
        for row in result.get(key, []) or []:
            if row.get("id"):
                provenance.add_paper(row["id"], row.get("title", ""))
    card = result.get("card")
    if isinstance(card, dict) and card.get("paper_id"):
        provenance.add_paper(card["paper_id"], card.get("title", ""))
    for row in result.get("contradictions", []) or []:
        for pid_key, ev_key in (("paper_a", "ev_a"), ("paper_b", "ev_b")):
            if row.get(pid_key):
                provenance.add_paper(row[pid_key], "", row.get(ev_key))


def citations_to_markdown(citations: list[Citation]) -> str:
    """Render a citation list as a Markdown reference block."""
    lines = []
    for c in citations:
        loc = f" ({c.evidence_location})" if c.evidence_location else ""
        lines.append(f"[{c.marker}] {c.title}{loc}")
    return "\n".join(lines)
