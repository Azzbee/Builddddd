"""Semantic Scholar Academic Graph client.

Provides paper metadata, citation counts, references (for bibliographic coupling),
and precomputed SPECTER2 vectors (``embedding.specter_v2``), which means papers
already indexed by S2 need no local embedding inference.
"""

from __future__ import annotations

from typing import Any

from lattice.config import EnrichmentSettings
from lattice.core.hashing import normalize_arxiv, normalize_doi
from lattice.enrichment.base import CachedHTTPClient
from lattice.enrichment.models import EnrichedPaper, ExternalAuthor

_FIELDS = (
    "paperId,title,year,venue,citationCount,referenceCount,externalIds,"
    "embedding.specter_v2,references.paperId,references.externalIds,"
    "authors.authorId,authors.name"
)


def paper_id_param(*, doi: str | None = None, arxiv_id: str | None = None, s2_id: str | None = None) -> str:
    """Build the S2 paper identifier path segment from any available id."""
    if s2_id:
        return s2_id
    doi = normalize_doi(doi)
    if doi:
        return f"DOI:{doi}"
    arxiv_id = normalize_arxiv(arxiv_id)
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"
    raise ValueError("need at least one of doi, arxiv_id, s2_id")


def parse_s2_paper(data: dict[str, Any]) -> EnrichedPaper:
    """Map a raw S2 paper JSON object to an EnrichedPaper. Pure and tested."""
    ext = data.get("externalIds") or {}
    embedding = (data.get("embedding") or {}).get("vector")
    ref_ids: list[str] = []
    for ref in data.get("references") or []:
        rid = ref.get("paperId")
        if rid:
            ref_ids.append(rid)
        else:
            rext = ref.get("externalIds") or {}
            if rext.get("DOI"):
                ref_ids.append(f"DOI:{normalize_doi(rext['DOI'])}")
    authors = [
        ExternalAuthor(name=a.get("name", ""), s2_id=a.get("authorId"))
        for a in data.get("authors") or []
        if a.get("name")
    ]
    return EnrichedPaper(
        source="semantic_scholar",
        external_id=data.get("paperId"),
        s2_paper_id=data.get("paperId"),
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        title=data.get("title"),
        year=data.get("year"),
        venue=data.get("venue"),
        citation_count=data.get("citationCount"),
        reference_count=data.get("referenceCount"),
        reference_ids=ref_ids,
        authors=authors,
        specter_embedding=embedding,
    )


class SemanticScholarClient:
    def __init__(self, settings: EnrichmentSettings, http: CachedHTTPClient | None = None):
        self._settings = settings
        headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}
        self._http = http or CachedHTTPClient(
            cache_ttl_s=settings.cache_ttl_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            default_headers=headers,
        )

    async def get_paper(
        self, *, doi: str | None = None, arxiv_id: str | None = None, s2_id: str | None = None
    ) -> EnrichedPaper:
        ident = paper_id_param(doi=doi, arxiv_id=arxiv_id, s2_id=s2_id)
        url = f"{self._settings.s2_base_url}/paper/{ident}"
        data = await self._http.get_json(url, params={"fields": _FIELDS})
        return parse_s2_paper(data)
