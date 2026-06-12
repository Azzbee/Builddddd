"""Weekly delta digest: what changed in the graph this period.

Pure assembly of a structured report plus a Markdown rendering (for email/UI). The
delta inputs are gathered by the scheduler from Postgres/Neo4j; this module just
shapes and renders them so it is testable without any datastore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice.landscape.momentum import MomentumScore


@dataclass
class NewPaper:
    paper_id: str
    title: str
    year: int | None = None


@dataclass
class WeightChange:
    source_id: str
    target_id: str
    old_weight: float | None
    new_weight: float


@dataclass
class Contradiction:
    claim_a: str
    paper_a: str
    claim_b: str
    paper_b: str


@dataclass
class DigestInput:
    period_label: str
    new_papers: list[NewPaper] = field(default_factory=list)
    new_edges: int = 0
    weight_changes: list[WeightChange] = field(default_factory=list)
    community_shifts: int = 0
    contradictions: list[Contradiction] = field(default_factory=list)
    movers: list[MomentumScore] = field(default_factory=list)


@dataclass
class DigestReport:
    period_label: str
    summary: dict[str, int]
    new_papers: list[NewPaper]
    notable_weight_changes: list[WeightChange]
    contradictions: list[Contradiction]
    movers: list[MomentumScore]

    def to_json(self) -> dict[str, object]:
        return {
            "period_label": self.period_label,
            "summary": self.summary,
            "new_papers": [{"id": p.paper_id, "title": p.title, "year": p.year} for p in self.new_papers],
            "notable_weight_changes": [
                {
                    "source": c.source_id,
                    "target": c.target_id,
                    "old": c.old_weight,
                    "new": round(c.new_weight, 4),
                }
                for c in self.notable_weight_changes
            ],
            "contradictions": [
                {"claim_a": c.claim_a, "paper_a": c.paper_a, "claim_b": c.claim_b, "paper_b": c.paper_b}
                for c in self.contradictions
            ],
            "movers": [m.to_json() for m in self.movers],
        }


def build_digest(
    data: DigestInput, *, weight_change_min: float = 0.1, max_movers: int = 5
) -> DigestReport:
    """Shape raw deltas into a ranked, thresholded report."""
    notable = [
        c
        for c in data.weight_changes
        if c.old_weight is None or abs(c.new_weight - c.old_weight) >= weight_change_min
    ]
    notable.sort(
        key=lambda c: abs(c.new_weight - (c.old_weight or 0.0)), reverse=True
    )
    movers = sorted(data.movers, key=lambda m: m.composite, reverse=True)[:max_movers]
    summary = {
        "new_papers": len(data.new_papers),
        "new_edges": data.new_edges,
        "weight_changes": len(notable),
        "community_shifts": data.community_shifts,
        "contradictions": len(data.contradictions),
    }
    return DigestReport(
        period_label=data.period_label,
        summary=summary,
        new_papers=data.new_papers,
        notable_weight_changes=notable,
        contradictions=data.contradictions,
        movers=movers,
    )


def render_markdown(report: DigestReport) -> str:
    """Render the digest as Markdown for email/UI. No em dashes (per house style)."""
    lines = [f"# Lattice digest: {report.period_label}", ""]
    s = report.summary
    lines.append(
        f"**{s['new_papers']}** new papers, **{s['new_edges']}** new edges, "
        f"**{s['weight_changes']}** notable weight changes, "
        f"**{s['contradictions']}** contradictions found."
    )
    lines.append("")

    if report.new_papers:
        lines.append("## New papers")
        for paper in report.new_papers:
            year = f" ({paper.year})" if paper.year else ""
            lines.append(f"- {paper.title}{year}")
        lines.append("")

    if report.movers:
        lines.append("## Movers")
        for m in report.movers:
            lines.append(
                f"- **{m.concept}**: momentum {m.composite:.2f}, {m.maturity}, "
                f"velocity {m.velocity:.1f}, acceleration {m.acceleration:.1f}"
            )
        lines.append("")

    if report.contradictions:
        lines.append("## Contradictions")
        for cont in report.contradictions:
            lines.append(f"- {cont.paper_a}: \"{cont.claim_a}\" vs {cont.paper_b}: \"{cont.claim_b}\"")
        lines.append("")

    if report.notable_weight_changes:
        lines.append("## Notable relationship changes")
        for wc in report.notable_weight_changes[:10]:
            old = f"{wc.old_weight:.2f}" if wc.old_weight is not None else "new"
            lines.append(f"- {wc.source_id} <-> {wc.target_id}: {old} -> {wc.new_weight:.2f}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
