"""OpenAlex client: open coverage, concepts, referenced works, and the global
aggregate statistics that the Landscape Intelligence module needs (publication
counts per concept per year). Uses the polite pool via a mailto parameter.
"""

from __future__ import annotations

from typing import Any

from lattice.config import EnrichmentSettings
from lattice.core.hashing import normalize_doi
from lattice.enrichment.base import CachedHTTPClient
from lattice.enrichment.models import EnrichedPaper, ExternalAuthor


def parse_openalex_work(data: dict[str, Any]) -> EnrichedPaper:
    """Map a raw OpenAlex work JSON object to an EnrichedPaper. Pure and tested."""
    ids = data.get("ids") or {}
    concepts = [c.get("display_name", "") for c in data.get("concepts") or [] if c.get("display_name")]
    authors = [
        ExternalAuthor(
            name=(a.get("author") or {}).get("display_name", ""),
            orcid=(a.get("author") or {}).get("orcid"),
        )
        for a in data.get("authorships") or []
        if (a.get("author") or {}).get("display_name")
    ]
    arxiv = None
    loc = data.get("primary_location") or {}
    src = loc.get("source") or {}
    return EnrichedPaper(
        source="openalex",
        external_id=data.get("id"),
        openalex_id=data.get("id"),
        doi=ids.get("doi") or data.get("doi"),
        arxiv_id=arxiv,
        title=data.get("title") or data.get("display_name"),
        year=data.get("publication_year"),
        venue=src.get("display_name"),
        citation_count=data.get("cited_by_count"),
        reference_count=len(data.get("referenced_works") or []),
        reference_ids=list(data.get("referenced_works") or []),
        concepts=concepts,
        authors=authors,
    )


class OpenAlexClient:
    def __init__(self, settings: EnrichmentSettings, http: CachedHTTPClient | None = None):
        self._settings = settings
        self._http = http or CachedHTTPClient(
            cache_ttl_s=settings.cache_ttl_s,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
        )

    @property
    def _mailto(self) -> dict[str, str]:
        return {"mailto": self._settings.openalex_mailto}

    async def get_work(self, *, doi: str | None = None, openalex_id: str | None = None) -> EnrichedPaper:
        if openalex_id:
            ident = openalex_id.rsplit("/", 1)[-1]
        elif doi:
            ident = f"doi:{normalize_doi(doi)}"
        else:
            raise ValueError("need doi or openalex_id")
        url = f"{self._settings.openalex_base_url}/works/{ident}"
        data = await self._http.get_json(url, params=self._mailto)
        return parse_openalex_work(data)

    async def concept_year_counts(self, concept: str, from_year: int, to_year: int) -> dict[int, int]:
        """Global publications per year matching a concept (for momentum/landscape).

        Returns ``{year: count}``. This is the "global signal" that distinguishes a
        true gap from a reading gap.
        """
        url = f"{self._settings.openalex_base_url}/works"
        params = {
            **self._mailto,
            "filter": f"concepts.search:{concept},from_publication_date:{from_year}-01-01,"
            f"to_publication_date:{to_year}-12-31",
            "group_by": "publication_year",
        }
        data = await self._http.get_json(url, params=params)
        out: dict[int, int] = {}
        for bucket in data.get("group_by") or []:
            try:
                out[int(bucket["key"])] = int(bucket["count"])
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def works_count(self, search: str) -> int:
        """Total global works matching a free-text search (for blind-spot detection)."""
        url = f"{self._settings.openalex_base_url}/works"
        params = {**self._mailto, "search": search, "per_page": "1"}
        data = await self._http.get_json(url, params=params)
        return int((data.get("meta") or {}).get("count", 0))
