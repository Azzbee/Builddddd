"""LLM-as-judge RAG evaluation (RAGAS-style, provider-agnostic).

The offline harness in :mod:`lattice.eval.retrieval_eval` uses cheap token-overlap
proxies so CI needs no API keys. This module adds the stronger, LLM-graded suite
for when a real provider is available:

* **faithfulness** — decompose the answer into atomic claims and check each is
  entailed by the retrieved context (hallucination detector).
* **answer relevance** — does the answer actually address the question.
* **context precision** — fraction of retrieved snippets that are relevant.
* **answer correctness** — agreement with the gold answer.

Every judge call goes through the same :class:`~lattice.core.llm.LLMClient`
abstraction as the rest of the system, so the harness is fully testable offline
with a scripted model (see tests/test_llm_judge.py) and portable across providers.
Abstentions are scored without a model call (faithful by construction, incorrect
by definition) to avoid wasting tokens and to match the offline harness' semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from lattice.core.llm import LLMClient, LLMMessage
from lattice.core.logging import get_logger
from lattice.eval.retrieval_eval import QAPair
from lattice.rag.models import AgentResult

log = get_logger("eval.llm_judge")


@dataclass
class JudgeVerdict:
    """A single graded dimension: a 0-1 score plus the judge's rationale/details."""

    score: float
    reasoning: str = ""
    details: dict[str, object] = field(default_factory=dict)


_FAITHFULNESS_PROMPT = """You are a strict fact-checker. Break the ANSWER into \
atomic factual claims, then decide for each whether it is supported by the CONTEXT \
(only the context — not your own knowledge).

Return ONLY JSON: {{"claims": [{{"claim": "...", "supported": true|false}}]}}.
If the answer makes no factual claims, return {{"claims": []}}.

CONTEXT:
{context}

ANSWER:
{answer}
"""

_RELEVANCE_PROMPT = """Rate how directly the ANSWER addresses the QUESTION, \
ignoring whether it is factually correct. 1.0 = fully on-point, 0.0 = irrelevant.

Return ONLY JSON: {{"score": 0.0-1.0, "reasoning": "..."}}.

QUESTION:
{question}

ANSWER:
{answer}
"""

_CONTEXT_PRECISION_PROMPT = """For each numbered CONTEXT snippet, decide whether it \
is relevant to answering the QUESTION.

Return ONLY JSON: {{"verdicts": [true|false, ...]}} with one entry per snippet, in order.

QUESTION:
{question}

CONTEXTS:
{contexts}
"""

_CORRECTNESS_PROMPT = """Compare the ANSWER to the reference GOLD answer. Score 1.0 \
if they agree on the key facts, 0.0 if they contradict or the answer misses the point.

Return ONLY JSON: {{"score": 0.0-1.0, "reasoning": "..."}}.

GOLD:
{gold}

ANSWER:
{answer}
"""


class LLMJudge:
    """Grades RAG answers with an LLM. One instance, four dimensions."""

    def __init__(self, llm: LLMClient, model: str = "claude-sonnet-4-6") -> None:
        self._llm = llm
        self._model = model

    async def _ask(self, prompt: str) -> dict[str, object]:
        resp = await self._llm.complete(
            self._model,
            [LLMMessage("user", prompt)],
            temperature=0.0,
            json_mode=True,
            max_tokens=1024,
        )
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    async def faithfulness(self, answer: str, contexts: list[str]) -> JudgeVerdict:
        if not answer.strip():
            return JudgeVerdict(0.0, "empty answer")
        ctx = "\n\n".join(f"- {c}" for c in contexts) or "(no context retrieved)"
        data = await self._ask(_FAITHFULNESS_PROMPT.format(context=ctx, answer=answer))
        claims = data.get("claims") or []
        if not isinstance(claims, list) or not claims:
            # No factual claims -> nothing to hallucinate.
            return JudgeVerdict(1.0, "no factual claims", {"n_claims": 0})
        supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
        score = supported / len(claims)
        return JudgeVerdict(
            score,
            f"{supported}/{len(claims)} claims supported",
            {"n_claims": len(claims), "supported": supported},
        )

    async def answer_relevance(self, question: str, answer: str) -> JudgeVerdict:
        if not answer.strip():
            return JudgeVerdict(0.0, "empty answer")
        data = await self._ask(_RELEVANCE_PROMPT.format(question=question, answer=answer))
        return JudgeVerdict(_clamp(data.get("score")), str(data.get("reasoning", "")))

    async def context_precision(self, question: str, contexts: list[str]) -> JudgeVerdict:
        if not contexts:
            return JudgeVerdict(0.0, "no context retrieved", {"n": 0})
        numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        data = await self._ask(
            _CONTEXT_PRECISION_PROMPT.format(question=question, contexts=numbered)
        )
        verdicts = data.get("verdicts") or []
        if not isinstance(verdicts, list) or not verdicts:
            return JudgeVerdict(0.0, "no verdicts", {"n": len(contexts)})
        relevant = sum(1 for v in verdicts if v)
        return JudgeVerdict(
            relevant / len(verdicts),
            f"{relevant}/{len(verdicts)} snippets relevant",
            {"n": len(verdicts), "relevant": relevant},
        )

    async def answer_correctness(self, answer: str, gold: str) -> JudgeVerdict:
        if not answer.strip():
            return JudgeVerdict(0.0, "empty answer")
        data = await self._ask(_CORRECTNESS_PROMPT.format(gold=gold, answer=answer))
        return JudgeVerdict(_clamp(data.get("score")), str(data.get("reasoning", "")))


def _clamp(value: object) -> float:
    if not isinstance(value, int | float | str):
        return 0.0
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class JudgedItem:
    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    answer_correctness: float
    citation_correctness: float
    abstained: bool

    def to_json(self) -> dict[str, object]:
        return {
            "question": self.question,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "citation_correctness": round(self.citation_correctness, 4),
            "abstained": self.abstained,
        }


@dataclass
class JudgedRagReport:
    """Aggregate LLM-judged RAG report over a QA set."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    answer_correctness: float
    citation_correctness: float
    abstention_rate: float
    n: int
    items: list[JudgedItem] = field(default_factory=list)

    def passes(
        self,
        *,
        faithfulness_min: float = 0.85,
        citation_min: float = 0.9,
        correctness_min: float = 0.7,
    ) -> bool:
        return (
            self.faithfulness >= faithfulness_min
            and self.citation_correctness >= citation_min
            and self.answer_correctness >= correctness_min
        )

    def to_json(self) -> dict[str, object]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "citation_correctness": round(self.citation_correctness, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "n": self.n,
            "items": [it.to_json() for it in self.items],
        }


def _contexts_from_result(result: AgentResult) -> list[str]:
    """Reconstruct the retrieved context for a result from its citation snippets."""
    out: list[str] = []
    for c in result.citations:
        if c.snippet:
            out.append(c.snippet)
        elif c.title:
            out.append(c.title)
    return out


async def evaluate_rag_judged(
    pairs: list[QAPair],
    results: list[AgentResult],
    judge: LLMJudge,
    *,
    contexts: list[list[str]] | None = None,
) -> JudgedRagReport:
    """Grade aligned (QAPair, AgentResult) pairs with an LLM judge.

    ``contexts`` optionally supplies the retrieved snippets per item (e.g. captured
    from the agent's tool calls); when omitted they are reconstructed from each
    result's citation snippets. Abstentions are scored without model calls.
    """
    if len(pairs) != len(results):
        raise ValueError("pairs and results must align")
    if contexts is not None and len(contexts) != len(pairs):
        raise ValueError("contexts must align with pairs")
    if not pairs:
        return JudgedRagReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    from lattice.eval.retrieval_eval import citation_correctness

    items: list[JudgedItem] = []
    for i, (qa, res) in enumerate(zip(pairs, results, strict=True)):
        ctx = contexts[i] if contexts is not None else _contexts_from_result(res)
        cite = citation_correctness(res, qa.relevant_paper_ids)
        if res.abstained:
            items.append(JudgedItem(qa.question, 1.0, 0.0, 0.0, 0.0, cite, abstained=True))
            continue
        faith = await judge.faithfulness(res.answer, ctx)
        rel = await judge.answer_relevance(qa.question, res.answer)
        prec = await judge.context_precision(qa.question, ctx)
        corr = await judge.answer_correctness(res.answer, qa.gold_answer)
        items.append(
            JudgedItem(
                qa.question,
                faith.score,
                rel.score,
                prec.score,
                corr.score,
                cite,
                abstained=False,
            )
        )

    n = len(items)

    def mean(attr: str) -> float:
        return float(sum(getattr(it, attr) for it in items)) / n

    report = JudgedRagReport(
        faithfulness=mean("faithfulness"),
        answer_relevance=mean("answer_relevance"),
        context_precision=mean("context_precision"),
        answer_correctness=mean("answer_correctness"),
        citation_correctness=mean("citation_correctness"),
        abstention_rate=sum(1 for it in items if it.abstained) / n,
        n=n,
        items=items,
    )
    log.info(
        "rag.judged",
        n=n,
        faithfulness=round(report.faithfulness, 3),
        correctness=round(report.answer_correctness, 3),
        abstention_rate=round(report.abstention_rate, 3),
    )
    return report
