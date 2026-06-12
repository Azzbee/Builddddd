"""arXiv watcher: fetch recent papers in watched categories.

The watcher only fetches and parses here; the similarity-to-corpus filter and the
human approval queue live in the evolution/watch layer (M7). The Atom-feed parser
is pure so it is tested against a fixture with no network.
"""

from __future__ import annotations

import re

from lattice.config import EnrichmentSettings
from lattice.core.hashing import normalize_arxiv
from lattice.enrichment.base import CachedHTTPClient
from lattice.enrichment.models import ArxivResult

_ATOM = "http://www.w3.org/2005/Atom"
_NS = {"a": _ATOM, "arxiv": "http://arxiv.org/schemas/atom"}
_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    return _WS.sub(" ", (text or "").strip())


def parse_arxiv_feed(xml: str | bytes) -> list[ArxivResult]:
    """Parse an arXiv Atom feed into ArxivResult records."""
    from lxml import etree

    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    root = etree.fromstring(xml)
    results: list[ArxivResult] = []
    for entry in root.findall("a:entry", _NS):
        raw_id = _clean(entry.findtext("a:id", default="", namespaces=_NS))
        arxiv_id = normalize_arxiv(raw_id)
        if not arxiv_id:
            continue
        title = _clean(entry.findtext("a:title", default="", namespaces=_NS))
        summary = _clean(entry.findtext("a:summary", default="", namespaces=_NS))
        published = _clean(entry.findtext("a:published", default="", namespaces=_NS)) or None
        authors = [
            _clean(name.text)
            for name in entry.findall("a:author/a:name", _NS)
            if name.text
        ]
        categories = [
            c.get("term", "") for c in entry.findall("a:category", _NS) if c.get("term")
        ]
        pdf_url = None
        for link in entry.findall("a:link", _NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        results.append(
            ArxivResult(
                arxiv_id=arxiv_id,
                title=title,
                summary=summary,
                authors=authors,
                categories=categories,
                published=published,
                pdf_url=pdf_url,
            )
        )
    return results


class ArxivWatcher:
    def __init__(self, settings: EnrichmentSettings, http: CachedHTTPClient | None = None):
        self._settings = settings
        # Short cache: we want fresh results, but still want to be polite on retries.
        self._http = http or CachedHTTPClient(
            cache_ttl_s=60 * 30,
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
        )

    async def fetch_recent(self, categories: list[str], max_results: int = 50) -> list[ArxivResult]:
        query = " OR ".join(f"cat:{c}" for c in categories)
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results),
        }
        # arXiv returns Atom XML, not JSON; fetch raw text via the cached client.
        url = self._settings.arxiv_base_url
        text = await self._http.get_text(url, params=params, cache_key=f"arxiv:{query}:{max_results}")
        return parse_arxiv_feed(text)
