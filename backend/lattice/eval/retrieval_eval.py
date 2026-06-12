"""RAG eval: faithfulness, answer relevance, and citation correctness.

A lightweight RAGAS-style harness. Citation correctness checks whether the cited
evidence actually belongs to a relevant paper. Faithfulness is approximated by
whether the answer's citations are all grounded (the synthesis guard enforces
this), with an optional injected LLM judge for stronger faithfulness scoring.
Answer relevance uses a token-overlap proxy or an injected judge.

Thresholds (PRD): faithfulness > 0.85, citation correctness > 0.9.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from lattice.eval.metrics import token_f1
from lattice.rag.models import AgentResult

# Optional judges return a 0-1 score for (answer, reference/context).
Judge = Callable[[str, str], float]


@dataclass
class QAPair:
    question: str
    gold_answer: str
    relevant_paper_ids: set[str] = field(default_factory=set)


@dataclass
class RetrievalReport:
    faithfulness: float
    answer_relevance: float
    citation_correctness: float
    abstention_rate: float
    n: int

    def passes(self, *, faithfulness_min: float = 0.85, citation_min: float = 0.9) -> bool:
        return self.faithfulness >= faithfulness_min and self.citation_correctness >= citation_min

    def to_json(self) -> dict[str, object]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "citation_correctness": round(self.citation_correctness, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "n": self.n,
        }


def citation_correctness(result: AgentResult, relevant: set[str]) -> float:
    """Fraction of cited papers that are actually relevant to the question."""
    if not result.citations:
        return 1.0 if not relevant else 0.0
    correct = sum(1 for c in result.citations if c.paper_id in relevant)
    return correct / len(result.citations)


def faithfulness(result: AgentResult, judge: Judge | None = None) -> float:
    """Are the answer's claims grounded? Proxy: cited + grounded; judge if provided.

    Abstentions are faithful by construction (the agent declined to assert).
    """
    if result.abstained:
        return 1.0
    if not result.citations:
        return 0.0  # asserted something with no grounding
    if judge is not None:
        context = " ".join(c.snippet or c.title for c in result.citations)
        return judge(result.answer, context)
    return 1.0  # synthesis guarantees citations resolve to retrieved provenance


def evaluate_rag(
    pairs: list[QAPair],
    results: list[AgentResult],
    *,
    relevance_judge: Judge | None = None,
    faithfulness_judge: Judge | None = None,
) -> RetrievalReport:
    if len(pairs) != len(results):
        raise ValueError("pairs and results must align")
    if not pairs:
        return RetrievalReport(0.0, 0.0, 0.0, 0.0, 0)

    faiths, rels, cites = [], [], []
    abstentions = 0
    for qa, res in zip(pairs, results, strict=True):
        faiths.append(faithfulness(res, faithfulness_judge))
        cites.append(citation_correctness(res, qa.relevant_paper_ids))
        if relevance_judge is not None:
            rels.append(relevance_judge(res.answer, qa.gold_answer))
        else:
            rels.append(token_f1(res.answer, qa.gold_answer))
        if res.abstained:
            abstentions += 1

    n = len(pairs)
    return RetrievalReport(
        faithfulness=sum(faiths) / n,
        answer_relevance=sum(rels) / n,
        citation_correctness=sum(cites) / n,
        abstention_rate=abstentions / n,
        n=n,
    )
