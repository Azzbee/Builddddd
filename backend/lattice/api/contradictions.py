from __future__ import annotations

from fastapi import APIRouter, Depends

from lattice.api.deps import Container, get_container, require_auth
from lattice.graph.contradictions import HeuristicNLIJudge, LLMNLIJudge

router = APIRouter(
    prefix="/contradictions", tags=["contradictions"], dependencies=[Depends(require_auth)]
)


@router.post("/analyze")
async def analyze(
    use_llm: bool = False, c: Container = Depends(get_container)
) -> dict[str, object]:
    """Run claim-level relation detection over the corpus and persist edges.

    Uses the deterministic heuristic judge by default; ``use_llm=true`` switches to
    the LLM NLI judge (needs provider keys).
    """
    judge = LLMNLIJudge(c.llm) if use_llm else HeuristicNLIJudge()
    edges = await c.ingestion.analyze_relations(judge)
    contradictions = [e for e in edges if str(e.relation) == "CONTRADICTS"]
    supports = [e for e in edges if str(e.relation) == "SUPPORTS"]
    return {
        "analyzed": len(edges),
        "contradictions": len(contradictions),
        "supports": len(supports),
        "edges": [e.to_json() for e in edges],
    }


@router.get("")
async def list_relations(
    relation: str | None = None, c: Container = Depends(get_container)
) -> list[dict[str, object]]:
    """Return claim relations, optionally filtered (e.g. relation=CONTRADICTS)."""
    edges = await c.ingestion.get_claim_relations(relation)
    return [e.to_json() for e in edges]
