"""Related-work draft generator with BibTeX export.

Given papers grouped by graph community, produce a structured related-work section:
one subsection per cluster, summarizing the methods and datasets, citing every
paper, and explicitly hedging where the evidence is thin (small or single-author
clusters). The draft is deterministic and grounded by construction (it cites only
the papers given); an optional LLM pass can polish the prose, but the offline path
is genuinely usable.

BibTeX keys are stable (firstauthor + year + first title word).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from lattice.extraction.schemas import PaperCard

_WORD = re.compile(r"[a-zA-Z]+")
_STOP_TITLE = {"a", "an", "the", "on", "of", "for", "and", "to", "in", "with"}


def bibtex_key(card: PaperCard) -> str:
    author = "anon"
    if card.authors:
        surname = card.authors[0].name.strip().split()
        if surname:
            author = re.sub(r"[^a-z]", "", surname[-1].lower()) or "anon"
    year = str(card.year) if card.year else "nd"
    first_word = "untitled"
    for w in _WORD.findall(card.title.lower()):
        if w not in _STOP_TITLE:
            first_word = w
            break
    return f"{author}{year}{first_word}"


def _bibtex_authors(card: PaperCard) -> str:
    return " and ".join(a.name for a in card.authors) if card.authors else "Unknown"


def to_bibtex(card: PaperCard) -> str:
    key = bibtex_key(card)
    entry_type = "article" if card.venue else "misc"
    lines = [f"@{entry_type}{{{key},", f"  title = {{{card.title}}},", f"  author = {{{_bibtex_authors(card)}}},"]
    if card.year:
        lines.append(f"  year = {{{card.year}}},")
    if card.venue:
        lines.append(f"  journal = {{{card.venue}}},")
    if card.doi:
        lines.append(f"  doi = {{{card.doi}}},")
    if card.arxiv_id:
        lines.append(f"  eprint = {{{card.arxiv_id}}},")
        lines.append("  archivePrefix = {arXiv},")
    lines.append("}")
    return "\n".join(lines)


def corpus_to_bibtex(cards: list[PaperCard]) -> str:
    seen: set[str] = set()
    entries: list[str] = []
    for card in sorted(cards, key=bibtex_key):
        key = bibtex_key(card)
        if key in seen:
            continue
        seen.add(key)
        entries.append(to_bibtex(card))
    return "\n\n".join(entries) + ("\n" if entries else "")


@dataclass
class RelatedWorkCluster:
    label: str
    cards: list[PaperCard]
    summary: str
    hedged: bool

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "summary": self.summary,
            "hedged": self.hedged,
            "papers": [{"paper_id": c.paper_id, "key": bibtex_key(c), "title": c.title} for c in self.cards],
        }


@dataclass
class RelatedWorkDraft:
    clusters: list[RelatedWorkCluster] = field(default_factory=list)

    def markdown(self) -> str:
        lines = ["# Related work", ""]
        for cl in self.clusters:
            lines.append(f"## {cl.label}")
            lines.append("")
            lines.append(cl.summary)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def bibtex(self) -> str:
        all_cards = [c for cl in self.clusters for c in cl.cards]
        return corpus_to_bibtex(all_cards)


def _top(counter: Counter[str], n: int) -> list[str]:
    return [k for k, _ in counter.most_common(n) if k]


def _cluster_summary(cards: list[PaperCard]) -> tuple[str, bool]:
    methods: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    for c in cards:
        methods.update(m for m in (c.methods_taxonomy + c.methodology.techniques) if m.strip())
        datasets.update(d.name for d in c.datasets if d.name.strip())
    cites = ", ".join(f"[@{bibtex_key(c)}]" for c in cards)
    method_str = ", ".join(_top(methods, 4)) or "a range of methods"
    data_str = ", ".join(_top(datasets, 3))
    hedged = len(cards) < 2
    parts = [f"A line of work {cites} investigates this area using {method_str}."]
    if data_str:
        parts.append(f"Common evaluation data includes {data_str}.")
    if hedged:
        parts.append(
            "Evidence here is thin (a single paper in the corpus), so conclusions "
            "should be treated as preliminary."
        )
    return " ".join(parts), hedged


def build_related_work(
    groups: dict[str, list[PaperCard]], *, label_prefix: str = "Cluster"
) -> RelatedWorkDraft:
    """Build a deterministic, grounded related-work draft from clustered papers."""
    clusters: list[RelatedWorkCluster] = []
    # Largest clusters first; label by dominant domain when available.
    for idx, (key, cards) in enumerate(
        sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True), start=1
    ):
        domains: Counter[str] = Counter()
        for c in cards:
            domains.update(d for d in c.domains if d.strip())
        label = _top(domains, 1)
        summary, hedged = _cluster_summary(cards)
        clusters.append(
            RelatedWorkCluster(
                label=label[0].title() if label else f"{label_prefix} {idx} (key {key})",
                cards=cards,
                summary=summary,
                hedged=hedged,
            )
        )
    return RelatedWorkDraft(clusters=clusters)
