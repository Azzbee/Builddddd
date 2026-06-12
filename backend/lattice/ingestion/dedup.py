"""Deduplication: re-ingesting the same paper is a no-op.

Four signals, in precedence order:
1. content hash (byte-identical PDF)
2. normalized DOI
3. normalized arXiv id (version-insensitive)
4. fuzzy title + author-overlap match (handles preprint vs published, reformats)

The :class:`CorpusIndex` is an in-memory lookup populated from Postgres at the
start of an ingest job. Keeping dedup as pure logic over an index makes it fully
testable without a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from lattice.core.hashing import normalize_arxiv, normalize_doi, normalize_text

#: rapidfuzz token_sort_ratio above this (0-100) counts as a title match.
TITLE_FUZZ_THRESHOLD = 92.0
#: Minimum fraction of shared normalized author surnames for a fuzzy dup.
AUTHOR_OVERLAP_THRESHOLD = 0.5


def author_surnames(authors: list[str]) -> set[str]:
    """Extract normalized surnames (last whitespace token) for overlap checks."""
    out: set[str] = set()
    for a in authors:
        norm = normalize_text(a)
        if norm:
            out.add(norm.split()[-1])
    return out


@dataclass(frozen=True)
class PaperIdentity:
    """The dedup-relevant fields of a candidate or existing paper."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    content_hash: str | None = None

    @property
    def norm_title(self) -> str:
        return normalize_text(self.title)

    @property
    def norm_doi(self) -> str | None:
        return normalize_doi(self.doi)

    @property
    def norm_arxiv(self) -> str | None:
        return normalize_arxiv(self.arxiv_id)


@dataclass
class DedupResult:
    is_duplicate: bool
    existing_paper_id: str | None = None
    reason: str | None = None


@dataclass
class CorpusIndex:
    """Fast lookup over existing papers, built once per job."""

    by_hash: dict[str, str] = field(default_factory=dict)
    by_doi: dict[str, str] = field(default_factory=dict)
    by_arxiv: dict[str, str] = field(default_factory=dict)
    titles: list[PaperIdentity] = field(default_factory=list)

    @classmethod
    def from_identities(cls, identities: list[PaperIdentity]) -> CorpusIndex:
        idx = cls()
        for ident in identities:
            idx.add(ident)
        return idx

    def add(self, ident: PaperIdentity) -> None:
        if ident.content_hash:
            self.by_hash[ident.content_hash] = ident.paper_id
        if ident.norm_doi:
            self.by_doi[ident.norm_doi] = ident.paper_id
        if ident.norm_arxiv:
            self.by_arxiv[ident.norm_arxiv] = ident.paper_id
        if ident.norm_title:
            self.titles.append(ident)

    def find_duplicate(self, candidate: PaperIdentity) -> DedupResult:
        if candidate.content_hash and candidate.content_hash in self.by_hash:
            return DedupResult(True, self.by_hash[candidate.content_hash], "content_hash")
        if candidate.norm_doi and candidate.norm_doi in self.by_doi:
            return DedupResult(True, self.by_doi[candidate.norm_doi], "doi")
        if candidate.norm_arxiv and candidate.norm_arxiv in self.by_arxiv:
            return DedupResult(True, self.by_arxiv[candidate.norm_arxiv], "arxiv")

        cand_title = candidate.norm_title
        if cand_title:
            cand_authors = author_surnames(candidate.authors)
            for existing in self.titles:
                score = fuzz.token_sort_ratio(cand_title, existing.norm_title)
                if score < TITLE_FUZZ_THRESHOLD:
                    continue
                if not cand_authors:
                    # No authors to corroborate; require a near-exact title.
                    if score >= 98.0:
                        return DedupResult(True, existing.paper_id, "title_exact")
                    continue
                ex_authors = author_surnames(existing.authors)
                if not ex_authors:
                    continue
                overlap = len(cand_authors & ex_authors) / len(cand_authors | ex_authors)
                if overlap >= AUTHOR_OVERLAP_THRESHOLD:
                    return DedupResult(True, existing.paper_id, "title_author_fuzzy")

        return DedupResult(False)
