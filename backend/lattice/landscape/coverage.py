"""Question-coverage probing: an auditable proxy for the corpus's unknown unknowns.

The epistemic quadrants (``landscape/quadrants.py``) deliberately leave the fourth
quadrant empty, because you cannot enumerate what a field does not know it does not
know. What you *can* do is probe. Assemble a bank of questions the corpus ought to be
able to answer, measure how well retrieval actually answers each one, and report the
ones it cannot. A question the corpus itself poses and cannot answer is a known
unknown made concrete; a question nobody in the corpus even asked and that it also
cannot answer is the honest proxy for a blind spot.

Coverage is scored from three normalized, auditable components, in the same spirit as
the edge-weight function:

    coverage(q) = w_r*R + w_s*S + w_t*T

    R  retrieval strength - the best fused hybrid-search score for the probe
    S  support breadth    - distinct papers contributing near-top evidence
    T  term grounding     - probe content tokens actually present in that evidence

``T`` carries the most weight because it is the only component free of embedding-
backend calibration: the hashing fallback used in demo/dev/CI and a real bge-m3 in
production produce very different score scales, but "does this term appear in the
evidence" is the same question either way. Every component and the terms the corpus
could not ground are returned, so a blind spot is inspectable rather than asserted.

Everything here is pure over injected data, so it is fully testable offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from lattice.core.hashing import normalize_text
from lattice.landscape.matrix import MatrixCell
from lattice.landscape.quadrants import OpenProblemCluster

#: Default size of the probe bank. Generation is capped so a large facet cross
#: cannot explode quadratically; what the cap dropped is always reported.
DEFAULT_PROBE_LIMIT = 48

#: Component weights. Grounding dominates because it is calibration-free.
W_RETRIEVAL = 0.35
W_SUPPORT = 0.25
W_GROUNDING = 0.40

#: A hit counts as support when it scores within this fraction of the best hit.
SUPPORT_RATIO = 0.7
#: Distinct supporting papers at which breadth saturates.
SUPPORT_TARGET = 3
#: Supporting hits whose text is searched for the probe's terms.
GROUNDING_HITS = 5

COVERED_MIN = 0.60
PARTIAL_MIN = 0.35

#: Repeat mentions at which a probe's salience saturates.
SALIENCE_TARGET = 3
#: Minimum content tokens for a probe to be scorable ("How does it work?" is not).
MIN_PROBE_TOKENS = 2
#: Concurrent retrievals while probing. Below the default Postgres pool_max (10)
#: so a probe run cannot starve the rest of the API of connections.
PROBE_CONCURRENCY = 8


class ProbeSource(StrEnum):
    """Where a probe came from. Determines how to read an uncovered result."""

    #: A question a paper in the corpus explicitly poses.
    RESEARCH_QUESTION = "research_question"
    #: An open problem the corpus raises in future work.
    OPEN_PROBLEM = "open_problem"
    #: A facet crossing nobody in the corpus asked about (the blind-spot probe).
    FACET_CROSS = "facet_cross"


class CoverageState(StrEnum):
    COVERED = "covered"
    PARTIAL = "partial"
    UNCOVERED = "uncovered"


#: How strongly an *uncovered* probe of each origin is evidence of a blind spot.
#: A crossing nobody asked about that also has no answer is the only one of the
#: three that is a genuine unknown-unknown proxy. The other two are known unknowns
#: made concrete: the field already flagged them, and the quadrants view already
#: shows them. They stay in the ranking, heavily discounted, because an open
#: problem with *zero* corpus evidence is still worth seeing next to the rest.
SOURCE_WEIGHT: dict[ProbeSource, float] = {
    ProbeSource.FACET_CROSS: 1.0,
    ProbeSource.OPEN_PROBLEM: 0.45,
    ProbeSource.RESEARCH_QUESTION: 0.35,
}

# --------------------------------------------------------------------------- tokens
#: Interrogatives, auxiliaries and prose glue carry no retrieval signal, so they are
#: excluded from term grounding. Kept separate from the claim stopwords in
#: quadrants.py: probes are questions, claims are assertions.
_PROBE_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "on",
    "in",
    "with",
    "to",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "whether",
    "any",
    "some",
    "there",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "their",
    "than",
    "then",
    "over",
    "under",
    "about",
    "into",
    "from",
    "by",
    "as",
    "at",
    "more",
    "most",
    "well",
    "using",
    "use",
    "used",
    "across",
    "within",
    "between",
    "during",
    "through",
    "other",
    "others",
    "such",
    "also",
    "based",
    "via",
    "per",
    "both",
    "each",
    "while",
    "given",
    "much",
    "many",
    "study",
    "studies",
    "work",
    "paper",
    "papers",
    "known",
    "know",
    "show",
    "shows",
    "compare",
    "compared",
    "perform",
    "performs",
    "relate",
    "relates",
    "reveal",
    "reveals",
    "address",
    "addresses",
    "apply",
    "applied",
    "transfer",
    "results",
    "result",
}


def term_forms(text: str) -> dict[str, str]:
    """Map each matching stem to the word it came from.

    Grounding matches on de-pluralized stems, but the stems are not words
    ("commodities" -> "commoditie"), and ``missing_terms`` is read by a human. So
    the stem is what matches and the original word is what is reported.
    """
    forms: dict[str, str] = {}
    for raw in normalize_text(text).split():
        if len(raw) < 3 or raw in _PROBE_STOP:
            continue
        stem = raw[:-1] if raw.endswith("s") and len(raw) > 3 else raw
        forms.setdefault(stem, raw)
    return forms


def content_tokens(text: str) -> set[str]:
    """Stopworded, de-pluralized content stems (len>=3) used for term grounding."""
    return set(term_forms(text))


# --------------------------------------------------------------------------- probes
@dataclass(frozen=True)
class Probe:
    """One question put to the corpus."""

    text: str
    source: ProbeSource
    #: How heavily the corpus leans on this question, normalized to [0, 1].
    salience: float = 0.0
    #: Papers that raised it (empty for templated facet crossings).
    origin_paper_ids: tuple[str, ...] = ()
    #: The (row, col) facet values a FACET_CROSS probe is about, for deep-linking
    #: straight into the proposal generator.
    facet_cell: tuple[str, str] | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "source": str(self.source),
            "salience": round(self.salience, 3),
            "origin_paper_ids": list(self.origin_paper_ids),
            "facet_cell": list(self.facet_cell) if self.facet_cell else None,
        }


@dataclass
class QuestionMention:
    """A research question as one paper stated it."""

    paper_id: str
    text: str


@dataclass
class ProbeBank:
    """The generated probes plus an honest account of what the cap dropped."""

    probes: list[Probe] = field(default_factory=list)
    generated: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return any(self.dropped.values())


#: Question templates per facet pair. Reversed pairs reuse the same template with
#: the values swapped, so only one direction is spelled out.
_FACET_TEMPLATES: dict[tuple[str, str], str] = {
    ("method", "dataset"): "How does {row} perform on {col}?",
    ("method", "concept"): "How well does {row} address {col}?",
    ("dataset", "concept"): "What does {row} reveal about {col}?",
    ("method", "method"): "How does {row} compare with {col}?",
    ("dataset", "dataset"): "How do results on {row} transfer to {col}?",
    ("concept", "concept"): "How does {row} relate to {col}?",
}


def facet_question(row_facet: str, col_facet: str, row: str, col: str) -> str:
    """Render the probe question for one gap-matrix cell."""
    template = _FACET_TEMPLATES.get((row_facet, col_facet))
    if template is not None:
        return template.format(row=row, col=col)
    template = _FACET_TEMPLATES.get((col_facet, row_facet))
    if template is not None:
        return template.format(row=col, col=row)
    return f"What is known about {row} and {col}?"


def _salience(count: int) -> float:
    return min(count / SALIENCE_TARGET, 1.0)


def _probes_from_questions(mentions: Sequence[QuestionMention]) -> list[Probe]:
    """One probe per distinct research question, salient by how often it recurs."""
    grouped: dict[str, tuple[str, list[str]]] = {}
    for m in mentions:
        text = m.text.strip()
        key = normalize_text(text)
        if not key or len(content_tokens(text)) < MIN_PROBE_TOKENS:
            continue
        if key not in grouped:
            grouped[key] = (text, [])
        papers = grouped[key][1]
        if m.paper_id not in papers:
            papers.append(m.paper_id)
    probes = [
        Probe(
            text=text,
            source=ProbeSource.RESEARCH_QUESTION,
            salience=_salience(len(papers)),
            origin_paper_ids=tuple(papers),
        )
        for text, papers in grouped.values()
    ]
    probes.sort(key=lambda p: (-p.salience, p.text))
    return probes


def _probes_from_open_problems(clusters: Sequence[OpenProblemCluster]) -> list[Probe]:
    """One probe per clustered open problem, salient by how many papers raised it."""
    probes: list[Probe] = []
    seen: set[str] = set()
    for cl in clusters:
        text = cl.canonical_text.strip()
        key = normalize_text(text)
        if not key or key in seen or len(content_tokens(text)) < MIN_PROBE_TOKENS:
            continue
        seen.add(key)
        probes.append(
            Probe(
                text=text,
                source=ProbeSource.OPEN_PROBLEM,
                salience=_salience(cl.frequency),
                origin_paper_ids=tuple(dict.fromkeys(cl.paper_ids)),
            )
        )
    probes.sort(key=lambda p: (-p.salience, p.text))
    return probes


def _probes_from_gaps(cells: Sequence[MatrixCell], row_facet: str, col_facet: str) -> list[Probe]:
    """One probe per gap cell, salient by its gap score relative to the top gap.

    ``gap_score`` is a product of three heuristics (feasibility x adjacency x
    demand) whose absolute magnitude means nothing on its own: a typical corpus
    scores every cell in the low tenths. Rescaling against the strongest gap makes
    it comparable with the mention-count salience the other two sources use, so
    pressure ranks probes across sources rather than by which scale happened to be
    larger.
    """
    top = max((c.gap_score for c in cells), default=0.0)
    probes: list[Probe] = []
    seen: set[str] = set()
    for cell in cells:
        text = facet_question(row_facet, col_facet, cell.row, cell.col)
        key = normalize_text(text)
        if not key or key in seen or len(content_tokens(text)) < MIN_PROBE_TOKENS:
            continue
        seen.add(key)
        probes.append(
            Probe(
                text=text,
                source=ProbeSource.FACET_CROSS,
                salience=min(max(cell.gap_score, 0.0) / top, 1.0) if top > 0 else 0.0,
                facet_cell=(cell.row, cell.col),
            )
        )
    probes.sort(key=lambda p: (-p.salience, p.text))
    return probes


def generate_probes(
    questions: Sequence[QuestionMention] = (),
    open_problems: Sequence[OpenProblemCluster] = (),
    gap_cells: Sequence[MatrixCell] = (),
    *,
    row_facet: str = "method",
    col_facet: str = "dataset",
    limit: int = DEFAULT_PROBE_LIMIT,
) -> ProbeBank:
    """Assemble the probe bank from all three sources, fairly capped at ``limit``.

    Sources are interleaved round-robin (each already ranked by salience) so one
    prolific source cannot crowd the others out of the bank, and whatever the cap
    dropped is reported per source rather than silently sampled away.
    """
    by_source = [
        _probes_from_questions(questions),
        _probes_from_open_problems(open_problems),
        _probes_from_gaps(gap_cells, row_facet, col_facet),
    ]
    generated = {str(lst[0].source): len(lst) for lst in by_source if lst}
    limit = max(limit, 0)

    kept: list[Probe] = []
    seen: set[str] = set()
    index = 0
    while len(kept) < limit and any(index < len(lst) for lst in by_source):
        for lst in by_source:
            if index >= len(lst) or len(kept) >= limit:
                continue
            probe = lst[index]
            key = normalize_text(probe.text)
            if key in seen:
                continue
            seen.add(key)
            kept.append(probe)
        index += 1

    taken = {str(source): 0 for source in ProbeSource}
    for probe in kept:
        taken[str(probe.source)] += 1
    dropped = {src: count - taken.get(src, 0) for src, count in generated.items()}
    return ProbeBank(
        probes=kept,
        generated=generated,
        dropped={src: n for src, n in dropped.items() if n > 0},
    )


# --------------------------------------------------------------------------- scoring
@dataclass
class ProbeEvidence:
    """One retrieved chunk offered as evidence for a probe."""

    paper_id: str
    title: str
    text: str
    score: float
    section: str | None = None
    evidence_location: str | None = None
    page: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "evidence_location": self.evidence_location,
            "page": self.page,
            "score": round(self.score, 4),
        }


@dataclass
class ProbeResult:
    """A scored probe: the verdict, its three components, and what was missing."""

    probe: Probe
    coverage: float
    state: CoverageState
    retrieval: float
    support: float
    grounding: float
    pressure: float
    supporting_papers: list[str]
    missing_terms: list[str]
    best_evidence: ProbeEvidence | None = None

    def to_json(self) -> dict[str, object]:
        return {
            **self.probe.to_json(),
            "coverage": round(self.coverage, 4),
            "state": str(self.state),
            "components": {
                "retrieval": round(self.retrieval, 3),
                "support": round(self.support, 3),
                "grounding": round(self.grounding, 3),
            },
            "pressure": round(self.pressure, 4),
            "supporting_papers": self.supporting_papers,
            "missing_terms": self.missing_terms,
            "best_evidence": self.best_evidence.to_json() if self.best_evidence else None,
        }


def _classify(coverage: float) -> CoverageState:
    if coverage >= COVERED_MIN:
        return CoverageState.COVERED
    if coverage >= PARTIAL_MIN:
        return CoverageState.PARTIAL
    return CoverageState.UNCOVERED


def score_probe(
    probe: Probe,
    evidence: Sequence[ProbeEvidence],
    *,
    support_ratio: float = SUPPORT_RATIO,
    support_target: int = SUPPORT_TARGET,
    grounding_hits: int = GROUNDING_HITS,
) -> ProbeResult:
    """Score one probe against the evidence retrieval returned for it."""
    forms = term_forms(probe.text)
    tokens = set(forms)
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)
    if not ranked or ranked[0].score <= 0:
        return ProbeResult(
            probe=probe,
            coverage=0.0,
            state=CoverageState.UNCOVERED,
            retrieval=0.0,
            support=0.0,
            grounding=0.0,
            pressure=SOURCE_WEIGHT[probe.source] * (0.5 + 0.5 * probe.salience),
            supporting_papers=[],
            missing_terms=sorted(forms.values()),
            best_evidence=None,
        )

    top = ranked[0]
    retrieval = min(max(top.score, 0.0), 1.0)

    floor = top.score * support_ratio
    supporting = [e for e in ranked if e.score >= floor]
    papers = list(dict.fromkeys(e.paper_id for e in supporting))
    support = min(len(papers) / support_target, 1.0) if support_target > 0 else 0.0

    grounded: set[str] = set()
    for e in supporting[:grounding_hits]:
        grounded |= content_tokens(f"{e.title} {e.section or ''} {e.text}")
    hit_tokens = tokens & grounded
    grounding = len(hit_tokens) / len(tokens) if tokens else 0.0

    coverage = W_RETRIEVAL * retrieval + W_SUPPORT * support + W_GROUNDING * grounding
    pressure = (1.0 - coverage) * SOURCE_WEIGHT[probe.source] * (0.5 + 0.5 * probe.salience)
    return ProbeResult(
        probe=probe,
        coverage=coverage,
        state=_classify(coverage),
        retrieval=retrieval,
        support=support,
        grounding=grounding,
        pressure=pressure,
        supporting_papers=papers,
        missing_terms=sorted(forms[stem] for stem in tokens - grounded),
        best_evidence=top,
    )


def blind_spots(results: Sequence[ProbeResult], limit: int = 10) -> list[ProbeResult]:
    """Probes the corpus cannot fully answer, highest pressure first."""
    candidates = [r for r in results if r.state is not CoverageState.COVERED]
    candidates.sort(key=lambda r: (r.pressure, r.probe.text), reverse=True)
    return candidates[:limit]


def summarize_coverage(results: Sequence[ProbeResult]) -> dict[str, object]:
    """Corpus-level coverage index plus state and source histograms."""
    states = {str(s): 0 for s in CoverageState}
    sources = {str(s): 0 for s in ProbeSource}
    for r in results:
        states[str(r.state)] += 1
        sources[str(r.probe.source)] += 1
    index = sum(r.coverage for r in results) / len(results) if results else 0.0
    uncovered = states[str(CoverageState.UNCOVERED)]
    return {
        "probe_count": len(results),
        "coverage_index": round(index, 4),
        "blind_spot_ratio": round(uncovered / len(results), 4) if results else 0.0,
        "by_state": states,
        "by_source": sources,
    }
