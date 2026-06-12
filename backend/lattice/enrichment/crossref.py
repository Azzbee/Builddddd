"""Crossref client: authoritative DOI resolution and metadata."""

from __future__ import annotations

from typing import Any

from lattice.config import EnrichmentSettings
from lattice.core.hashing import normalize_doi
from lattice.enrichment.base import CachedHTTPClient
from lattice.enrichment.models import EnrichedPaper, ExternalAuthor


def parse_crossref_work(data: dict[str, Any]) -> EnrichedPaper:
    msg = data.get("message", data)
    title_list = msg.get("title") or []
    title = title_list[0] if title_list else None
    container = msg.get("container-title") or []
    venue = container[0] if container else None
    year = None
    issued = (msg.get("issued") or {}).get("date-parts") or []
    if issued and issued[0]:
        year = issued[0][0]
    authors = [
        ExternalAuthor(
            name=" ".join(p for p in [a.get("given"), a.get("family")] if p),
            orcid=a.get("ORCID"),
        )
        for a in msg.get("author") or []
        if a.get("family")
    ]
    return EnrichedPaper(
        source="crossref",
        external_id=msg.get("DOI"),
        doi=msg.get("DOI"),
        title=title,
        year=year,
        venue=venue,
        citation_count=msg.get("is-referenced-by-count"),
        reference_count=msg.get("references-count"),
        authors=authors,
    )


class CrossrefClient:
    def __init__(self, settings: EnrichmentSettings, http: CachedHTTPClient | None = None):
        self._settings = settings
        self._http = http or CachedHTTPClient(
            cache_ttl_s=settings.cache_ttl_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            default_headers={"User-Agent": f"Lattice/0.1 (mailto:{settings.openalex_mailto})"},
        )

    async def get_by_doi(self, doi: str) -> EnrichedPaper:
        ndoi = normalize_doi(doi)
        url = f"{self._settings.crossref_base_url}/works/{ndoi}"
        data = await self._http.get_json(url)
        return parse_crossref_work(data)
