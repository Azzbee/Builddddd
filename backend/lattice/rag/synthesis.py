"""Cited-answer assembly.

The agent accumulates a provenance pool as tools return results. The final answer
declares which papers it cites; synthesis validates every claimed citation against
the provenance pool, dropping any paper that was never actually retrieved. This is
the citation-correctness guard: the agent cannot cite what it did not see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lattice.rag.models import ChunkHit, Citation


@dataclass
class Provenance:
    """Everything the agent has actually seen, keyed by paper id."""

    papers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_chunk(self, hit: ChunkHit) -> None:
        entry = self.papers.setdefault(
            hit.paper_id, {"title": hit.title, "sections": set(), "snippet": None, "evidence": None}
        )
        entry["sections"].add(hit.section_title)
        if entry["snippet"] is None:
            entry["snippet"] = hit.text[:280]
            entry["evidence"] = hit.evidence_location or hit.section_title

    def add_paper(self, paper_id: str, title: str, evidence: str | None = None) -> None:
        entry = self.papers.setdefault(
            paper_id, {"title": title, "sections": set(), "snippet": None, "evidence": evidence}
        )
        entry["title"] = entry.get("title") or title

    def has(self, paper_id: str) -> bool:
        return paper_id in self.papers


def build_citations(claimed: list[dict[str, Any]], provenance: Provenance) -> list[Citation]:
    """Map claimed citations to grounded Citation objects, dropping ungrounded ones."""
    citations: list[Citation] = []
    marker = 1
    seen: set[str] = set()
    for c in claimed:
        pid = str(c.get("paper_id", ""))
        if not pid or pid in seen or not provenance.has(pid):
            continue  # citation-correctness: cannot cite unseen papers
        seen.add(pid)
        entry = provenance.papers[pid]
        citations.append(
            Citation(
                marker=marker,
                paper_id=pid,
                title=entry["title"],
                section=c.get("section") or next(iter(entry["sections"]), None),
                evidence_location=c.get("evidence_location") or entry.get("evidence"),
                snippet=entry.get("snippet"),
            )
        )
        marker += 1
    return citations
