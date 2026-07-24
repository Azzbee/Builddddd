from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from lattice.api.deps import Container, get_container, require_auth
from lattice.core.logging import get_logger
from lattice.rag.models import AgentResult

log = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_auth)])


class QueryRequest(BaseModel):
    question: str


@router.post("/stream")
async def query_stream(
    req: QueryRequest, c: Container = Depends(get_container)
) -> EventSourceResponse:
    """Stream the agent's reasoning and answer as Server-Sent Events."""
    agent = c.make_agent()

    async def gen() -> AsyncIterator[dict[str, str]]:
        saw_final = False
        try:
            async for event in agent.stream(req.question):
                data = event.data
                if event.type == "final":
                    result = data["result"]
                    assert isinstance(result, AgentResult)
                    payload = result.to_json()
                    saw_final = True
                else:
                    payload = data
                yield {"event": event.type, "data": json.dumps(payload, default=str)}
        except Exception:
            # The agent raised before yielding a final answer (provider outage,
            # cost cap, a bug). The SSE contract is that the stream always ends
            # with exactly one `final` event, so the client resets its UI instead
            # of hanging on "thinking...". Log the cause; do not leak it downstream.
            log.exception("query.stream failed")
        if not saw_final:
            fallback = AgentResult.failure(
                "The agent could not complete this query. Please try again."
            )
            yield {"event": "final", "data": json.dumps(fallback.to_json(), default=str)}

    return EventSourceResponse(gen())


@router.post("")
async def query(req: QueryRequest, c: Container = Depends(get_container)) -> dict[str, object]:
    """Non-streaming query (returns the final AgentResult)."""
    agent = c.make_agent()
    result = await agent.run(req.question)
    return result.to_json()
