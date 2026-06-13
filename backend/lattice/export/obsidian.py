"""Obsidian/Markdown export: one note per paper with wiki-links mirroring the
graph edges, so the knowledge graph lives in the user's PKM too.

Each note has YAML frontmatter (metadata), the structured card body, and a
"Related" section of ``[[wiki links]]`` to neighbor notes, weighted. Pure: takes a
card plus its neighbor (title, weight) list and returns Markdown.
"""

from __future__ import annotations

from lattice.extraction.schemas import PaperCard


def _yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def card_to_note(card: PaperCard, neighbors: list[tuple[str, float]]) -> str:
    """Render a PaperCard as an Obsidian note with wiki-linked neighbors."""
    fm = [
        "---",
        f'title: "{card.title}"',
        f"year: {card.year if card.year is not None else 'null'}",
        f"type: {card.paper_type}",
        f"doi: {card.doi or 'null'}",
        f"arxiv: {card.arxiv_id or 'null'}",
        f"confidence: {card.confidence}",
        f"authors: {_yaml_list([a.name for a in card.authors])}",
        f"methods: {_yaml_list(card.methods_taxonomy)}",
        f"domains: {_yaml_list(card.domains)}",
        "tags: [lattice, paper]",
        "---",
        "",
    ]
    body = [f"# {card.title}", ""]
    if card.problem_statement:
        body += ["## Problem", card.problem_statement, ""]
    body += ["## Methodology", card.methodology.approach_summary or "n/a", ""]
    if card.datasets:
        body += ["## Datasets", *[f"- {d.name}" for d in card.datasets], ""]
    if card.key_results:
        body += ["## Key results"]
        for r in card.key_results:
            loc = f" ({r.evidence_location})" if r.evidence_location else ""
            body.append(f"- {r.claim}{loc}")
        body.append("")
    if card.limitations:
        body += ["## Limitations", *[f"- {x}" for x in card.limitations], ""]
    if card.contributions:
        body += ["## Contributions", *[f"- {x}" for x in card.contributions], ""]

    related = sorted(neighbors, key=lambda n: n[1], reverse=True)
    if related:
        body += ["## Related"]
        for title, weight in related:
            body.append(f"- [[{title}]] (weight {weight:.2f})")
        body.append("")

    return "\n".join(fm + body).rstrip() + "\n"
