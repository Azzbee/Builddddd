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


def _suffix(i: int) -> str:
    """Spreadsheet-style lowercase suffix for the i-th colliding key (i >= 1):
    a, b, ..., z, aa, ab, ..., az, ba, ...

    i == 0 gets no suffix, so the base key stays canonical. Handles arbitrarily many
    collisions, unlike ``chr(ord('a') + i)`` which overflows past 'z' at i == 26.
    """
    s = ""
    while i > 0:
        i, rem = divmod(i - 1, 26)
        s = chr(ord("a") + rem) + s
    return s


def assign_unique_keys(cards: list[PaperCard]) -> dict[str, str]:
    """Map each paper_id to a citation key that is unique across ``cards``.

    Two distinct papers can generate the same base key (same first-author surname +
    year + first title word), e.g. two 2020 Zhang papers whose titles both start
    with "Forecasting". Silently deduping by key would drop one paper from the
    bibliography and misattribute its [@key] marker. Instead, disambiguate colliding
    keys deterministically (zhang2020forecasting, zhang2020forecastinga, ...), ordered
    by paper_id so the assignment is stable regardless of input order. A suffixed key
    that would clash with an already-assigned key is skipped, so the result is always
    globally unique.
    """
    # Group paper_ids by base key. Dedup identical paper_ids (the same card can
    # appear in multiple clusters) so they share one entry.
    groups: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for card in cards:
        if card.paper_id in seen_ids:
            continue
        seen_ids.add(card.paper_id)
        groups.setdefault(bibtex_key(card), []).append(card.paper_id)

    keys: dict[str, str] = {}
    used: set[str] = set()
    # Process base keys in sorted order for a fully deterministic assignment.
    for base in sorted(groups):
        for pid in sorted(groups[base]):
            i = 0
            candidate = base
            # Advance the suffix until we land on a key nobody else has taken (guards
            # against a suffixed key colliding with another paper's literal base key).
            while candidate in used:
                i += 1
                candidate = f"{base}{_suffix(i)}"
            used.add(candidate)
            keys[pid] = candidate
    return keys


def _bibtex_authors(card: PaperCard) -> str:
    return " and ".join(a.name for a in card.authors) if card.authors else "Unknown"


def to_bibtex(card: PaperCard, key: str | None = None) -> str:
    key = key or bibtex_key(card)
    entry_type = "article" if card.venue else "misc"
    lines = [
        f"@{entry_type}{{{key},",
        f"  title = {{{card.title}}},",
        f"  author = {{{_bibtex_authors(card)}}},",
    ]
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
    """Render every distinct paper to a BibTeX entry with a collision-free key.

    Distinct papers that collide on their base key are disambiguated (not dropped),
    so no paper silently disappears from the bibliography.
    """
    keys = assign_unique_keys(cards)
    seen_ids: set[str] = set()
    entries: list[tuple[str, str]] = []
    for card in cards:
        if card.paper_id in seen_ids:
            continue  # same paper across clusters -> one entry
        seen_ids.add(card.paper_id)
        key = keys[card.paper_id]
        entries.append((key, to_bibtex(card, key)))
    entries.sort(key=lambda e: e[0])
    return "\n\n".join(e for _k, e in entries) + ("\n" if entries else "")


@dataclass
class RelatedWorkCluster:
    label: str
    cards: list[PaperCard]
    summary: str
    hedged: bool
    keys: dict[str, str] = field(default_factory=dict)  # paper_id -> unique cite key

    def _key(self, card: PaperCard) -> str:
        return self.keys.get(card.paper_id) or bibtex_key(card)

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "summary": self.summary,
            "hedged": self.hedged,
            "papers": [
                {"paper_id": c.paper_id, "key": self._key(c), "title": c.title} for c in self.cards
            ],
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


def _cluster_summary(
    cards: list[PaperCard], keys: dict[str, str] | None = None
) -> tuple[str, bool]:
    keys = keys or {}
    methods: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    for c in cards:
        methods.update(m for m in (c.methods_taxonomy + c.methodology.techniques) if m.strip())
        datasets.update(d.name for d in c.datasets if d.name.strip())
    cites = ", ".join(f"[@{keys.get(c.paper_id) or bibtex_key(c)}]" for c in cards)
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
    # One corpus-wide key map so cite markers, the papers list, and the BibTeX all
    # agree even when two distinct papers collide on their base key.
    keys = assign_unique_keys([c for cards in groups.values() for c in cards])
    # Largest clusters first; label by dominant domain when available.
    for idx, (key, cards) in enumerate(
        sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True), start=1
    ):
        domains: Counter[str] = Counter()
        for c in cards:
            domains.update(d for d in c.domains if d.strip())
        label = _top(domains, 1)
        summary, hedged = _cluster_summary(cards, keys)
        clusters.append(
            RelatedWorkCluster(
                label=label[0].title() if label else f"{label_prefix} {idx} (key {key})",
                cards=cards,
                summary=summary,
                hedged=hedged,
                keys=keys,
            )
        )
    return RelatedWorkDraft(clusters=clusters)
