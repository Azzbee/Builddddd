"""Typed results from enrichment sources, normalized across providers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lattice.core.hashing import normalize_arxiv, normalize_doi


class ExternalAuthor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    s2_id: str | None = None
    orcid: str | None = None


class EnrichedPaper(BaseModel):
    """A paper record assembled from one or more external sources."""

    model_config = ConfigDict(extra="ignore")

    source: str  # "semantic_scholar" | "openalex" | "crossref"
    external_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    s2_paper_id: str | None = None
    openalex_id: str | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    #: External ids of papers this one cites (for bibliographic coupling).
    reference_ids: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    authors: list[ExternalAuthor] = Field(default_factory=list)
    specter_embedding: list[float] | None = None

    def model_post_init(self, _ctx: object) -> None:
        object.__setattr__(self, "doi", normalize_doi(self.doi))
        object.__setattr__(self, "arxiv_id", normalize_arxiv(self.arxiv_id))


class ArxivResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    arxiv_id: str
    title: str
    summary: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: str | None = None
    pdf_url: str | None = None
