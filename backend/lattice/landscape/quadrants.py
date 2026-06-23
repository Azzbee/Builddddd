"""Epistemic Quadrants (the Rumsfeld view). Each quadrant is a computable query.

* Known knowns: claims supported by >=2 *independent* papers (no shared authors),
  no active contradiction, bonus for triangulation across methods/datasets.
* Known unknowns: open problems clustered from future_work/limitations, ranked by
  frequency x recency.
* Unknown knowns: cross-community transfer candidates (an answer next door) and
  buried evidence (the literature forgot something).
* Unknown unknowns: proxies only, honestly labeled (interstices, high-pressure
  empty cells, question-coverage probing) - left to higher layers.

These functions are pure over injected data so they are fully testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lattice.core.hashing import normalize_text
from lattice.graph.similarity import cosine

# --------------------------------------------------------------------------- claim clustering
_CLAIM_STOP = {
    "the", "a", "an", "of", "for", "on", "in", "with", "to", "and", "or", "is",
    "are", "than", "over", "versus", "vs", "that", "this", "these", "show", "shows",
    "showed", "we", "our", "it", "its", "by", "at", "as", "be", "can", "using",
    "significantly", "results", "result", "improvement", "improvements",
}


def _claim_tokens(text: str) -> set[str]:
    """Content tokens for fuzzy claim matching: stopworded, de-pluralized, len>=3."""
    toks: set[str] = set()
    for raw in normalize_text(text).split():
        if len(raw) < 3 or raw in _CLAIM_STOP:
            continue
        toks.add(raw[:-1] if raw.endswith("s") and len(raw) > 3 else raw)
    return toks


@dataclass
class ClaimCluster:
    canonical: str
    members: list[str]
    papers: list[SupportingPaper]
    polarity: int = 1


def cluster_claims(
    items: list[tuple[str, SupportingPaper]], *, threshold: float = 0.5
) -> list[ClaimCluster]:
    """Greedily cluster claims that state the same finding in different words.

    Uses the overlap coefficient |A n B| / min(|A|, |B|) on content tokens, which is
    robust to differing verbosity (e.g. "LSTM significantly improves accuracy over
    ARIMA" vs "GRU improves forecasting accuracy versus ARIMA"). Exact-text grouping
    misses these and reports zero established findings on real corpora. Claims of
    opposite polarity never merge, so "X improves Y" and "X does not improve Y" stay
    in separate clusters rather than being conflated.
    """
    from lattice.graph.contradictions import claim_polarity

    clusters: list[tuple[set[str], ClaimCluster]] = []
    for claim, sp in items:
        toks = _claim_tokens(claim)
        if not toks:
            continue
        pol = claim_polarity(claim) or 1  # treat neutral as positive for grouping
        best: tuple[set[str], ClaimCluster] | None = None
        best_sim = 0.0
        for ctoks, cl in clusters:
            if cl.polarity != pol:
                continue  # never merge a claim with its negation
            inter = len(toks & ctoks)
            sim = inter / min(len(toks), len(ctoks)) if inter else 0.0
            if sim > best_sim:
                best_sim, best = sim, (ctoks, cl)
        if best is not None and best_sim >= threshold:
            ctoks, cl = best
            ctoks |= toks
            cl.members.append(claim)
            cl.papers.append(sp)
            # Keep the most descriptive (longest) claim as the canonical label.
            if len(claim) > len(cl.canonical):
                cl.canonical = claim
        else:
            clusters.append(
                (set(toks), ClaimCluster(canonical=claim, members=[claim], papers=[sp], polarity=pol))
            )
    return [cl for _toks, cl in clusters]


# --------------------------------------------------------------------------- known knowns
@dataclass
class SupportingPaper:
    paper_id: str
    authors: frozenset[str]
    method: str | None = None
    dataset: str | None = None
    year: int | None = None


@dataclass
class ConsolidatedFinding:
    claim: str
    support_count: int
    independent_supports: int
    triangulated: bool  # spans >=2 methods or datasets
    latest_year: int | None
    strength: float


def _independent_groups(papers: list[SupportingPaper]) -> int:
    """Count author-disjoint groups (a proxy for independent confirmation)."""
    groups: list[set[str]] = []
    for p in papers:
        placed = False
        for g in groups:
            if g & p.authors:
                g |= set(p.authors)
                placed = True
                break
        if not placed:
            groups.append(set(p.authors))
    return len(groups)


def consolidate_known_knowns(
    claims: dict[str, list[SupportingPaper]],
    contradicted: set[str] | None = None,
    *,
    min_independent: int = 2,
) -> list[ConsolidatedFinding]:
    """Return established findings, strongest first."""
    contradicted = contradicted or set()
    findings: list[ConsolidatedFinding] = []
    for claim, papers in claims.items():
        if claim in contradicted or not papers:
            continue
        independent = _independent_groups(papers)
        if independent < min_independent:
            continue
        methods = {p.method for p in papers if p.method}
        datasets = {p.dataset for p in papers if p.dataset}
        triangulated = len(methods) >= 2 or len(datasets) >= 2
        latest = max((p.year for p in papers if p.year is not None), default=None)
        strength = independent + (0.5 if triangulated else 0.0)
        findings.append(
            ConsolidatedFinding(
                claim=claim,
                support_count=len(papers),
                independent_supports=independent,
                triangulated=triangulated,
                latest_year=latest,
                strength=strength,
            )
        )
    findings.sort(key=lambda f: (f.strength, f.support_count), reverse=True)
    return findings


# --------------------------------------------------------------------------- known unknowns
@dataclass
class OpenProblemMention:
    paper_id: str
    text: str
    embedding: list[float]
    year: int | None = None


@dataclass
class OpenProblemCluster:
    canonical_text: str
    paper_ids: list[str]
    frequency: int
    latest_year: int | None
    score: float
    span_years: int = 0


@dataclass
class _Cluster:
    centroid: np.ndarray
    mentions: list[OpenProblemMention]
    canonical: str


def cluster_open_problems(
    mentions: list[OpenProblemMention],
    *,
    threshold: float = 0.75,
    current_year: int | None = None,
) -> list[OpenProblemCluster]:
    """Greedily cluster open-problem mentions by embedding cosine similarity.

    Each cluster becomes a canonical OpenProblem ranked by frequency x recency. A
    problem raised repeatedly over many years is flagged via ``span_years``.
    """
    clusters: list[_Cluster] = []
    for m in mentions:
        vec = np.asarray(m.embedding, dtype=float)
        best: _Cluster | None = None
        best_sim = 0.0
        for cl in clusters:
            sim = cosine(vec, cl.centroid)
            if sim is not None and sim > best_sim:
                best_sim, best = sim, cl
        if best is not None and best_sim >= threshold:
            best.mentions.append(m)
            n = len(best.mentions)
            best.centroid = (best.centroid * (n - 1) + vec) / n  # running mean
        else:
            clusters.append(_Cluster(centroid=vec, mentions=[m], canonical=m.text))

    out: list[OpenProblemCluster] = []
    for cl in clusters:
        ms = cl.mentions
        years = [m.year for m in ms if m.year is not None]
        latest = max(years, default=None)
        span = (max(years) - min(years)) if len(years) >= 2 else 0
        recency = 1.0
        if current_year is not None and latest is not None:
            recency = 1.0 / (1.0 + max(0, current_year - latest))
        score = len(ms) * recency
        out.append(
            OpenProblemCluster(
                canonical_text=cl.canonical,
                paper_ids=[m.paper_id for m in ms],
                frequency=len(ms),
                latest_year=latest,
                score=score,
                span_years=span,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


# --------------------------------------------------------------------------- unknown knowns
@dataclass
class TransferCandidate:
    method: str
    source_paper: str
    target_paper: str
    similarity: float
    graph_distance: int


def cross_community_transfer(
    source_methods: list[tuple[str, str, list[float]]],  # (method, paper_id, methodology_emb)
    target_problems: list[tuple[str, list[float]]],  # (paper_id, problem_emb)
    distance_fn: Callable[[str, str], int],
    *,
    sim_threshold: float = 0.78,
    min_distance: int = 3,
) -> list[TransferCandidate]:
    """Find 'the answer exists next door' candidates.

    High similarity between a community's methodology vectors and another's problem
    vectors, but far apart in the graph (no citation/RELATED path).
    """
    candidates: list[TransferCandidate] = []
    for method, src_paper, meth_emb in source_methods:
        mv = np.asarray(meth_emb, dtype=float)
        for tgt_paper, prob_emb in target_problems:
            if src_paper == tgt_paper:
                continue
            sim = cosine(mv, np.asarray(prob_emb, dtype=float))
            if sim is None or sim < sim_threshold:
                continue
            dist = distance_fn(src_paper, tgt_paper)
            if dist > min_distance:
                candidates.append(
                    TransferCandidate(method, src_paper, tgt_paper, sim, dist)
                )
    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return candidates
