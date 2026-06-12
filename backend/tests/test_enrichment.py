from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from lattice.config import EnrichmentSettings
from lattice.core.errors import NotFoundError, RateLimitedError
from lattice.enrichment.arxiv_watcher import parse_arxiv_feed
from lattice.enrichment.base import CachedHTTPClient, InMemoryCache
from lattice.enrichment.crossref import parse_crossref_work
from lattice.enrichment.openalex import OpenAlexClient, parse_openalex_work
from lattice.enrichment.semantic_scholar import (
    SemanticScholarClient,
    paper_id_param,
    parse_s2_paper,
)

pytest.importorskip("lxml")
FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- pure parsers
def test_parse_s2_paper() -> None:
    data = {
        "paperId": "abc123",
        "title": "Copper forecasting",
        "year": 2024,
        "venue": "JCF",
        "citationCount": 12,
        "referenceCount": 2,
        "externalIds": {"DOI": "10.1/X", "ArXiv": "2401.00001"},
        "embedding": {"model": "specter_v2", "vector": [0.1, 0.2, 0.3]},
        "references": [{"paperId": "ref1"}, {"externalIds": {"DOI": "10.2/Y"}}],
        "authors": [{"authorId": "a1", "name": "Jane Doe"}],
    }
    p = parse_s2_paper(data)
    assert p.s2_paper_id == "abc123"
    assert p.doi == "10.1/x"
    assert p.arxiv_id == "2401.00001"
    assert p.specter_embedding == [0.1, 0.2, 0.3]
    assert "ref1" in p.reference_ids
    assert any(r.startswith("DOI:10.2/y") for r in p.reference_ids)
    assert p.authors[0].name == "Jane Doe"


def test_parse_openalex_work() -> None:
    data = {
        "id": "https://openalex.org/W123",
        "ids": {"doi": "https://doi.org/10.1/Z"},
        "title": "A work",
        "publication_year": 2023,
        "cited_by_count": 5,
        "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
        "concepts": [{"display_name": "Forecasting"}, {"display_name": "Time series"}],
        "authorships": [{"author": {"display_name": "Carol", "orcid": "0000-1"}}],
        "primary_location": {"source": {"display_name": "Journal X"}},
    }
    p = parse_openalex_work(data)
    assert p.openalex_id == "https://openalex.org/W123"
    assert p.doi == "10.1/z"
    assert p.concepts == ["Forecasting", "Time series"]
    assert len(p.reference_ids) == 2
    assert p.venue == "Journal X"


def test_parse_crossref_work() -> None:
    data = {
        "message": {
            "DOI": "10.1/abc",
            "title": ["Crossref title"],
            "container-title": ["Some Journal"],
            "issued": {"date-parts": [[2022, 5, 1]]},
            "is-referenced-by-count": 7,
            "references-count": 30,
            "author": [{"given": "Jane", "family": "Doe", "ORCID": "0000-2"}],
        }
    }
    p = parse_crossref_work(data)
    assert p.doi == "10.1/abc"
    assert p.year == 2022
    assert p.venue == "Some Journal"
    assert p.authors[0].name == "Jane Doe"


def test_paper_id_param_precedence() -> None:
    assert paper_id_param(s2_id="S1") == "S1"
    assert paper_id_param(doi="https://doi.org/10.1/X") == "DOI:10.1/x"
    assert paper_id_param(arxiv_id="arXiv:2401.00001v2") == "ARXIV:2401.00001"
    with pytest.raises(ValueError):
        paper_id_param()


def test_parse_arxiv_feed() -> None:
    results = parse_arxiv_feed((FIXTURES / "arxiv_feed.xml").read_bytes())
    assert len(results) == 2
    first = results[0]
    assert first.arxiv_id == "2406.01234"  # version stripped
    assert "Phosphate" in first.title
    assert first.authors == ["Alice Researcher", "Bob Scientist"]
    assert "q-fin.ST" in first.categories
    assert first.pdf_url and first.pdf_url.endswith("2406.01234v1")


# --------------------------------------------------------------------------- HTTP client
@respx.mock
async def test_s2_client_fetch_and_cache() -> None:
    route = respx.get(url__regex=r".*/paper/DOI:10\.1/x").mock(
        return_value=httpx.Response(200, json={"paperId": "p", "title": "T", "year": 2024})
    )
    settings = EnrichmentSettings()
    async with httpx.AsyncClient() as ac:
        http = CachedHTTPClient(client=ac, cache=InMemoryCache())
        client = SemanticScholarClient(settings, http=http)
        p1 = await client.get_paper(doi="10.1/X")
        p2 = await client.get_paper(doi="10.1/X")  # served from cache
    assert p1.s2_paper_id == "p" and p2.title == "T"
    assert route.call_count == 1  # second call cached


@respx.mock
async def test_http_client_404_is_not_found() -> None:
    respx.get(url__regex=r".*/works/.*").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as ac:
        http = CachedHTTPClient(client=ac, cache=InMemoryCache(), backoff_base_s=0.001)
        client = OpenAlexClient(EnrichmentSettings(), http=http)
        with pytest.raises(NotFoundError):
            await client.get_work(doi="10.1/missing")


@respx.mock
async def test_http_client_retries_then_raises_rate_limited() -> None:
    route = respx.get(url__regex=r".*/works/.*").mock(return_value=httpx.Response(429))
    async with httpx.AsyncClient() as ac:
        http = CachedHTTPClient(
            client=ac, cache=InMemoryCache(), max_retries=3, backoff_base_s=0.001
        )
        client = OpenAlexClient(EnrichmentSettings(), http=http)
        with pytest.raises(RateLimitedError):
            await client.get_work(doi="10.1/x")
    assert route.call_count == 3  # retried up to max_retries


@respx.mock
async def test_openalex_concept_year_counts() -> None:
    respx.get(url__regex=r".*/works").mock(
        return_value=httpx.Response(
            200,
            json={"group_by": [{"key": "2022", "count": 10}, {"key": "2023", "count": 25}]},
        )
    )
    async with httpx.AsyncClient() as ac:
        http = CachedHTTPClient(client=ac, cache=InMemoryCache())
        client = OpenAlexClient(EnrichmentSettings(), http=http)
        counts = await client.concept_year_counts("forecasting", 2022, 2023)
    assert counts == {2022: 10, 2023: 25}
