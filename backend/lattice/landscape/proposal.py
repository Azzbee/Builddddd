"""Gap -> research-proposal generator.

Turns a gap-matrix cell (row facet x column facet, e.g. method x dataset) into a
structured, grounded research proposal:

* the **thesis** - what to try and why it is a gap,
* the **building blocks** already in the corpus - the row's track record and the
  column's track record, each cited,
* the **method to borrow** and where from,
* **why now** - momentum of the two areas plus the corpus's own demand signal,
* the **closest prior art** (cited), and
* the honest **risks**, including the Empty-vs-Blind-spot novelty framing.

Pure and deterministic over injected data, so it is fully testable offline and
grounded by construction - every referenced paper is a real corpus paper. An LLM
can later polish the prose, but the offline draft is genuinely usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice.core.hashing import normalize_text

#: A blind spot: the world has many papers on this cell but the corpus has none.
BLIND_SPOT_MIN_GLOBAL = 25


@dataclass
class FacetPaper:
    """A corpus paper projected onto the two facets under consideration."""

    paper_id: str
    title: str
    year: int | None
    row_values: set[str]
    col_values: set[str]
    representative_claim: str | None = None
    limitations: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    """A cited corpus paper backing a part of the proposal."""

    paper_id: str
    title: str
    year: int | None
    note: str

    def to_json(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "note": self.note,
        }


@dataclass
class MomentumLite:
    """The momentum facts the proposal needs (decoupled from MomentumScore)."""

    maturity: str
    composite: float
    burst: float


@dataclass
class ResearchProposal:
    row_facet: str
    col_facet: str
    row: str
    col: str
    state: str  # "gap" | "blind_spot" | "occupied"
    novelty: str
    gap_score: float
    components: dict[str, float]
    global_count: int
    demand: float
    thesis: str
    why_now: str
    row_track_record: list[Evidence]
    col_track_record: list[Evidence]
    method_to_borrow: Evidence | None
    recommended_baselines: list[str]
    prior_art: list[Evidence]
    risks: list[str]
    open_problems: list[str]
    confidence: float

    def to_json(self) -> dict[str, object]:
        return {
            "row_facet": self.row_facet,
            "col_facet": self.col_facet,
            "row": self.row,
            "col": self.col,
            "state": self.state,
            "novelty": self.novelty,
            "gap_score": round(self.gap_score, 4),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "global_count": self.global_count,
            "demand": round(self.demand, 3),
            "thesis": self.thesis,
            "why_now": self.why_now,
            "row_track_record": [e.to_json() for e in self.row_track_record],
            "col_track_record": [e.to_json() for e in self.col_track_record],
            "method_to_borrow": self.method_to_borrow.to_json() if self.method_to_borrow else None,
            "recommended_baselines": self.recommended_baselines,
            "prior_art": [e.to_json() for e in self.prior_art],
            "risks": self.risks,
            "open_problems": self.open_problems,
            "confidence": round(self.confidence, 3),
            "markdown": self.markdown(),
        }

    def markdown(self) -> str:
        lines = [
            f"# Research proposal: {self.row} x {self.col}",
            "",
            f"_{self.novelty}_",
            "",
            "## Thesis",
            self.thesis,
            "",
            "## Why now",
            self.why_now,
            "",
        ]
        if self.method_to_borrow:
            m = self.method_to_borrow
            lines += [
                "## Method to borrow",
                f"- **{m.note}** - {m.title} ({m.year or '?'})",
                "",
            ]
        if self.row_track_record:
            lines.append(f"## Track record for {self.row}")
            for e in self.row_track_record:
                lines.append(f"- {e.title} ({e.year or '?'}) - {e.note}")
            lines.append("")
        if self.col_track_record:
            lines.append(f"## Track record for {self.col}")
            for e in self.col_track_record:
                lines.append(f"- {e.title} ({e.year or '?'}) - {e.note}")
            lines.append("")
        if self.recommended_baselines:
            lines += [
                "## Baselines to beat",
                ", ".join(self.recommended_baselines),
                "",
            ]
        if self.open_problems:
            lines.append("## Open problems this would address")
            for p in self.open_problems:
                lines.append(f"- {p}")
            lines.append("")
        if self.risks:
            lines.append("## Risks")
            for r in self.risks:
                lines.append(f"- {r}")
            lines.append("")
        if self.prior_art:
            lines.append("## Prior art")
            for e in self.prior_art:
                lines.append(f"- {e.title} ({e.year or '?'})")
        return "\n".join(lines).rstrip() + "\n"


def _evidence(p: FacetPaper, note: str) -> Evidence:
    return Evidence(paper_id=p.paper_id, title=p.title, year=p.year, note=note)


def _dedup_limitations(papers: list[FacetPaper], limit: int = 4) -> list[str]:
    seen: dict[str, str] = {}
    for p in papers:
        for lim in p.limitations:
            key = normalize_text(lim)
            if key and key not in seen:
                seen[key] = lim.strip()
    return list(seen.values())[:limit]


def build_proposal(
    row_facet: str,
    col_facet: str,
    row: str,
    col: str,
    papers: list[FacetPaper],
    *,
    now_year: int,
    global_count: int = 0,
    demand: float = 0.0,
    row_momentum: MomentumLite | None = None,
    col_momentum: MomentumLite | None = None,
    open_problems: list[str] | None = None,
    method_to_borrow: Evidence | None = None,
    blind_spot_min_global: int = BLIND_SPOT_MIN_GLOBAL,
) -> ResearchProposal:
    """Compose a grounded proposal for the (row, col) cell from corpus evidence."""
    in_cell = [p for p in papers if row in p.row_values and col in p.col_values]
    row_side = [p for p in papers if row in p.row_values and p not in in_cell]
    col_side = [p for p in papers if col in p.col_values and p not in in_cell]

    # --- novelty / state framing -------------------------------------------------
    if in_cell:
        state = "occupied"
        novelty = (
            f"Already studied in the corpus ({len(in_cell)} paper(s)) - this maps the prior art."
        )
    elif global_count >= blind_spot_min_global:
        state = "blind_spot"
        novelty = (
            f"Blind spot: ~{global_count} papers exist worldwide but none in your corpus - "
            "likely tractable; validate novelty and harvest the external work first."
        )
    elif global_count > 0:
        state = "gap"
        novelty = (
            f"Lightly explored externally (~{global_count} papers) and absent from your corpus."
        )
    else:
        state = "gap"
        novelty = "Greenfield: no external work surfaced - higher risk, higher reward."

    # --- building blocks ---------------------------------------------------------
    row_other_cols = sorted({v for p in row_side for v in p.col_values})
    col_other_rows = sorted({v for p in col_side for v in p.row_values})

    row_track = [
        _evidence(p, f"applies {row} to {', '.join(sorted(p.col_values)) or 'related problems'}")
        for p in sorted(row_side, key=lambda p: p.year or 0, reverse=True)[:4]
    ]
    col_track = [
        _evidence(p, f"studies {col} with {', '.join(sorted(p.row_values)) or 'other methods'}")
        for p in sorted(col_side, key=lambda p: p.year or 0, reverse=True)[:4]
    ]

    # --- thesis ------------------------------------------------------------------
    if in_cell:
        thesis = (
            f"{row} has already been applied to {col}; extend or stress-test it "
            f"(see prior art) rather than treating it as unexplored."
        )
    else:
        clauses = [f"Apply **{row}** to **{col}**."]
        if row_other_cols:
            clauses.append(
                f"{row} is proven on {', '.join(row_other_cols[:3])}, suggesting it transfers."
            )
        if col_other_rows:
            clauses.append(
                f"{col} has so far only been tackled with {', '.join(col_other_rows[:3])}, "
                "leaving headroom for a stronger method."
            )
        if not row_side and not col_side:
            clauses.append("Neither side has corpus precedent, so scope a pilot before committing.")
        thesis = " ".join(clauses)

    # --- why now -----------------------------------------------------------------
    why_bits: list[str] = []
    if row_momentum:
        why_bits.append(f"{row} is {row_momentum.maturity} (momentum {row_momentum.composite:.2f})")
    if col_momentum:
        why_bits.append(f"{col} is {col_momentum.maturity} (momentum {col_momentum.composite:.2f})")
    if demand > 0.4:
        why_bits.append(f"the corpus's own future-work points here (demand {demand:.2f})")
    why_now = (
        "; ".join(why_bits).capitalize() + "."
        if why_bits
        else "Momentum is flat - justify timeliness from external trends before investing."
    )

    # --- baselines, risks, confidence -------------------------------------------
    baselines = col_other_rows[:5]
    risks: list[str] = []
    if not row_side:
        risks.append(f"No corpus track record for {row}: feasibility is unproven here.")
    if not col_side:
        risks.append(f"No corpus track record for {col}: the target may be ill-characterized.")
    if state == "blind_spot":
        risks.append("Heavily studied elsewhere: confirm you are not reinventing known results.")
    if state == "gap" and global_count == 0:
        risks.append(
            "No external signal found: the gap may reflect infeasibility, not opportunity."
        )
    risks.extend(f"Known pitfall: {lim}" for lim in _dedup_limitations(row_side + col_side))

    # Confidence in the *opportunity*: more building blocks + demand + momentum -> higher.
    block_strength = min(1.0, (len(row_side) + len(col_side)) / 6.0)
    momentum_strength = max(
        (row_momentum.composite if row_momentum else 0.0),
        (col_momentum.composite if col_momentum else 0.0),
    )
    confidence = round(0.5 * block_strength + 0.3 * demand + 0.2 * momentum_strength, 3)

    prior_art = _rank_prior_art(row_side, col_side)

    # The method to borrow is the row method itself; cite its strongest application.
    if method_to_borrow is None and row_side and not in_cell:
        best = max(row_side, key=lambda p: p.year or 0)
        method_to_borrow = _evidence(best, f"adopt {row}, demonstrated here")

    return ResearchProposal(
        row_facet=row_facet,
        col_facet=col_facet,
        row=row,
        col=col,
        state=state,
        novelty=novelty,
        gap_score=round(block_strength * max(demand, 0.0), 4),
        components={
            "building_blocks": round(block_strength, 3),
            "demand": round(demand, 3),
            "momentum": round(momentum_strength, 3),
        },
        global_count=global_count,
        demand=demand,
        thesis=thesis,
        why_now=why_now,
        row_track_record=row_track,
        col_track_record=col_track,
        method_to_borrow=method_to_borrow,
        recommended_baselines=baselines,
        prior_art=prior_art,
        risks=risks,
        open_problems=open_problems or [],
        confidence=confidence,
    )


def _rank_prior_art(
    row_side: list[FacetPaper], col_side: list[FacetPaper], limit: int = 6
) -> list[Evidence]:
    seen: dict[str, Evidence] = {}
    for p in sorted(row_side + col_side, key=lambda p: p.year or 0, reverse=True):
        if p.paper_id not in seen:
            seen[p.paper_id] = _evidence(p, "adjacent prior art")
    return list(seen.values())[:limit]
