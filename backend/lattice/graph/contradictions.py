"""Claim-level relation detection: contradictions, convergence, and extensions.

For pairs of claims that address the same concept, an NLI judge decides whether
they SUPPORT, CONTRADICT, or EXTEND each other. This is the single highest-wow
feature: it surfaces "Paper A reports X improves Y, Paper B finds no effect" as a
first-class CONTRADICTS edge with both evidence locations.

Two judges share one protocol:

* ``LLMNLIJudge`` - the production judge (provider-agnostic via LLMClient).
* ``HeuristicNLIJudge`` - a deterministic, offline judge using polarity and
  subject-overlap cues. It is a genuinely useful cheap prefilter and makes the
  whole pipeline testable with no model.

Candidate generation only compares claims that (a) share a concept and (b) are
topically similar, keeping the work O(pairs-within-concept) rather than O(n^2).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from lattice.core.hashing import normalize_text
from lattice.core.llm import LLMClient, LLMMessage


class ClaimRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    EXTENDS = "EXTENDS"
    UNRELATED = "UNRELATED"


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    paper_id: str
    text: str
    concept: str
    authors: frozenset[str] = frozenset()
    year: int | None = None
    evidence_location: str = ""


@dataclass
class ClaimEdge:
    source_id: str
    target_id: str
    source_paper: str
    target_paper: str
    relation: ClaimRelation
    confidence: float
    source_text: str = ""
    target_text: str = ""
    source_evidence: str = ""
    target_evidence: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_paper": self.source_paper,
            "target_paper": self.target_paper,
            "relation": str(self.relation),
            "confidence": round(self.confidence, 4),
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_evidence": self.source_evidence,
            "target_evidence": self.target_evidence,
        }


class NLIJudge(Protocol):
    async def judge(self, a: str, b: str) -> tuple[ClaimRelation, float]: ...


# ---------------------------------------------------------------------------
# Heuristic judge (deterministic, offline)
# ---------------------------------------------------------------------------
_POSITIVE = {
    "improves",
    "improve",
    "improved",
    "outperforms",
    "outperform",
    "beats",
    "beat",
    "better",
    "higher",
    "increase",
    "increases",
    "increased",
    "gains",
    "effective",
    "significant",
    "significantly",
    "superior",
    "boosts",
    "enhances",
    "reduces",
    "reduction",
    "lowers",
    "accurate",
    "wins",
    "exceeds",
    "positive",
}
_NEGATIVE = {
    "worse",
    "underperforms",
    "underperform",
    "decrease",
    "decreases",
    "decreased",
    "ineffective",
    "fails",
    "fail",
    "inferior",
    "degrades",
    "harms",
    "negative",
    "negligible",
    "insignificant",
    "weaker",
    "loses",
}
_NEGATORS = {"no", "not", "never", "without", "neither", "nor", "fails", "lacks", "cannot"}
_NULL_PHRASES = (
    "no effect",
    "no significant",
    "not significant",
    "no improvement",
    "does not",
    "did not",
    "fails to",
    "no benefit",
    "no difference",
)
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "on",
    "in",
    "to",
    "and",
    "or",
    "with",
    "is",
    "are",
    "was",
    "were",
    "we",
    "our",
    "that",
    "this",
    "than",
    "as",
    "by",
    "at",
    "it",
    "its",
    "from",
    "using",
    "use",
    "used",
    "show",
    "shows",
    "showed",
    "find",
    "finds",
    "found",
    "report",
    "reports",
    "results",
    "result",
    "method",
    "model",
    "models",
    "approach",
}
_WORD = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    toks = _WORD.findall(normalize_text(text))
    return {
        t
        for t in toks
        if t not in _STOP and t not in _POSITIVE and t not in _NEGATIVE and t not in _NEGATORS
    }


def claim_polarity(text: str) -> int:
    """Public alias: +1 asserts an effect, -1 negates it, 0 neutral. Used by the
    quadrant claim-clustering to avoid merging a claim with its negation."""
    return _polarity(text)


def _polarity(text: str) -> int:
    """Return +1 (asserts an effect), -1 (negates/null effect), or 0 (neutral)."""
    norm = normalize_text(text)
    if any(p in norm for p in _NULL_PHRASES):
        return -1
    toks = _WORD.findall(norm)
    score = 0
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i > 0 else ""
        if t in _POSITIVE:
            score += -1 if prev in _NEGATORS else 1
        elif t in _NEGATIVE:
            score += 1 if prev in _NEGATORS else -1
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def _subject_overlap(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class HeuristicNLIJudge:
    """Deterministic NLI via subject overlap + polarity. Offline, fast, explainable.

    ``subject_threshold`` of 0.5 requires the two claims to genuinely share their
    subject before a polarity clash counts as a contradiction; lower values produce
    false positives (e.g. "LSTM beats ARIMA" vs "transformers don't beat LSTM" share
    only generic tokens) that then suppress legitimate known-knowns downstream.
    """

    def __init__(self, subject_threshold: float = 0.5) -> None:
        self._threshold = subject_threshold

    async def judge(self, a: str, b: str) -> tuple[ClaimRelation, float]:
        overlap = _subject_overlap(a, b)
        if overlap < self._threshold:
            return ClaimRelation.UNRELATED, overlap
        pa, pb = _polarity(a), _polarity(b)
        if pa != 0 and pb != 0 and pa != pb:
            return ClaimRelation.CONTRADICTS, min(1.0, overlap + 0.2)
        if pa != 0 and pa == pb:
            ta, tb = _content_tokens(a), _content_tokens(b)
            # Same polarity but one strictly more specific -> EXTENDS.
            if ta < tb or tb < ta:
                return ClaimRelation.EXTENDS, overlap
            return ClaimRelation.SUPPORTS, min(1.0, overlap + 0.1)
        return ClaimRelation.UNRELATED, overlap


# ---------------------------------------------------------------------------
# LLM judge (production)
# ---------------------------------------------------------------------------
_NLI_PROMPT = """Two claims from different scientific papers address a related \
topic. Decide their logical relation.

Claim A: {a}
Claim B: {b}

Return ONLY JSON: {{"relation": "SUPPORTS|CONTRADICTS|EXTENDS|UNRELATED", \
"confidence": 0.0-1.0}}.
- SUPPORTS: both assert a consistent finding.
- CONTRADICTS: they assert opposing findings on the same question.
- EXTENDS: B refines or generalizes A (or vice versa) without conflict.
- UNRELATED: different questions.
Be conservative: only say CONTRADICTS when the findings genuinely conflict."""


class LLMNLIJudge:
    def __init__(self, llm: LLMClient, model: str = "claude-haiku-4-5-20251001") -> None:
        self._llm = llm
        self._model = model

    async def judge(self, a: str, b: str) -> tuple[ClaimRelation, float]:
        resp = await self._llm.complete(
            self._model,
            [LLMMessage("user", _NLI_PROMPT.format(a=a, b=b))],
            json_mode=True,
            max_tokens=128,
        )
        try:
            data = resp.json()
            relation = ClaimRelation(str(data["relation"]).upper())
            return relation, float(data.get("confidence", 0.5))
        except (json.JSONDecodeError, KeyError, ValueError):
            return ClaimRelation.UNRELATED, 0.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def candidate_pairs(claims: Iterable[ClaimRecord]) -> list[tuple[ClaimRecord, ClaimRecord]]:
    """Pairs of claims sharing a concept, from different papers. Independence
    (no shared authors) is preferred but not required at this stage."""
    by_concept: dict[str, list[ClaimRecord]] = {}
    for c in claims:
        by_concept.setdefault(normalize_text(c.concept), []).append(c)
    pairs: list[tuple[ClaimRecord, ClaimRecord]] = []
    for group in by_concept.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i].paper_id != group[j].paper_id:
                    pairs.append((group[i], group[j]))
    return pairs


@dataclass
class RelationReport:
    edges: list[ClaimEdge] = field(default_factory=list)

    @property
    def contradictions(self) -> list[ClaimEdge]:
        return [e for e in self.edges if e.relation == ClaimRelation.CONTRADICTS]

    @property
    def supports(self) -> list[ClaimEdge]:
        return [e for e in self.edges if e.relation == ClaimRelation.SUPPORTS]


def candidate_pairs_for(
    new_claims: Iterable[ClaimRecord], existing: Iterable[ClaimRecord]
) -> list[tuple[ClaimRecord, ClaimRecord]]:
    """Pairs of (new, existing) claims sharing a concept, from different papers.

    The incremental counterpart of :func:`candidate_pairs`: O(new x same-concept)
    instead of O(corpus^2), so relation detection can run on every ingest and the
    graph accumulates contradictions as papers arrive.
    """
    by_concept: dict[str, list[ClaimRecord]] = {}
    for c in existing:
        by_concept.setdefault(normalize_text(c.concept), []).append(c)
    pairs: list[tuple[ClaimRecord, ClaimRecord]] = []
    seen: set[tuple[str, str]] = set()
    for n in new_claims:
        for e in by_concept.get(normalize_text(n.concept), []):
            if e.paper_id == n.paper_id or (n.claim_id, e.claim_id) in seen:
                continue
            seen.add((n.claim_id, e.claim_id))
            pairs.append((n, e))
    return pairs


async def _judge_pairs(
    pairs: list[tuple[ClaimRecord, ClaimRecord]],
    judge: NLIJudge,
    min_confidence: float,
) -> RelationReport:
    edges: list[ClaimEdge] = []
    for a, b in pairs:
        relation, confidence = await judge.judge(a.text, b.text)
        if relation == ClaimRelation.UNRELATED or confidence < min_confidence:
            continue
        edges.append(
            ClaimEdge(
                source_id=a.claim_id,
                target_id=b.claim_id,
                source_paper=a.paper_id,
                target_paper=b.paper_id,
                relation=relation,
                confidence=confidence,
                source_text=a.text,
                target_text=b.text,
                source_evidence=a.evidence_location,
                target_evidence=b.evidence_location,
            )
        )
    return RelationReport(edges=edges)


async def detect_relations(
    claims: list[ClaimRecord],
    judge: NLIJudge,
    *,
    min_confidence: float = 0.5,
    max_pairs: int = 2000,
) -> RelationReport:
    """Run the judge over candidate pairs and collect non-trivial relations."""
    return await _judge_pairs(candidate_pairs(claims)[:max_pairs], judge, min_confidence)


async def detect_relations_for(
    new_claims: list[ClaimRecord],
    existing: list[ClaimRecord],
    judge: NLIJudge,
    *,
    min_confidence: float = 0.5,
    max_pairs: int = 500,
) -> RelationReport:
    """Incremental detection: one paper's claims against the existing corpus."""
    pairs = candidate_pairs_for(new_claims, existing)[:max_pairs]
    return await _judge_pairs(pairs, judge, min_confidence)
