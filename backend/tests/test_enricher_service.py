from __future__ import annotations

from lattice.config import EnrichmentSettings
from lattice.core.errors import NotFoundError
from lattice.enrichment.models import EnrichedPaper
from lattice.enrichment.service import CompositeEnricher
from lattice.extraction.schemas import Methodology, PaperCard


def _card(**kw) -> PaperCard:
    return PaperCard(paper_id="p1", title="t", methodology=Methodology(approach_summary="x"), **kw)


class FakeS2:
    def __init__(self, paper: EnrichedPaper | None = None, error: Exception | None = None) -> None:
        self.paper = paper
        self.error = error
        self.calls = 0

    async def get_paper(self, *, doi=None, arxiv_id=None, s2_id=None) -> EnrichedPaper:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.paper is not None
        return self.paper


class FakeOpenAlex:
    def __init__(self, work: EnrichedPaper | None = None, error: Exception | None = None) -> None:
        self.work = work
        self.error = error
        self.calls = 0

    async def get_work(self, *, doi=None, openalex_id=None) -> EnrichedPaper:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.work is not None
        return self.work


def _enricher(s2, oa) -> CompositeEnricher:
    e = CompositeEnricher(EnrichmentSettings())
    e._s2 = s2  # type: ignore[assignment]
    e._openalex = oa  # type: ignore[assignment]
    return e


async def test_enrich_uses_s2_specter_and_refs() -> None:
    s2 = FakeS2(
        EnrichedPaper(
            source="semantic_scholar",
            s2_paper_id="S1",
            reference_ids=["r1", "r2"],
            citation_count=12,
            specter_embedding=[0.1, 0.2],
        )
    )
    oa = FakeOpenAlex()
    out = await _enricher(s2, oa).enrich(_card(doi="10.1/x"))
    assert out["s2_paper_id"] == "S1"
    assert out["reference_ids"] == ["r1", "r2"]
    assert out["specter_embedding"] == [0.1, 0.2]
    assert oa.calls == 0  # S2 had references, no OpenAlex fallback needed


async def test_enrich_falls_back_to_openalex_for_refs() -> None:
    s2 = FakeS2(error=NotFoundError("not in s2"))
    oa = FakeOpenAlex(
        EnrichedPaper(source="openalex", reference_ids=["w1"], concepts=["Forecasting"])
    )
    out = await _enricher(s2, oa).enrich(_card(doi="10.1/x"))
    assert out["reference_ids"] == ["w1"]
    assert out["concepts"] == ["Forecasting"]
    assert oa.calls == 1


async def test_enrich_no_ids_skips_s2() -> None:
    s2 = FakeS2(error=AssertionError("should not be called"))
    out = await _enricher(s2, FakeOpenAlex()).enrich(_card())  # no doi/arxiv/s2
    assert out == {}
    assert s2.calls == 0


async def test_enrich_all_fail_returns_empty() -> None:
    s2 = FakeS2(error=NotFoundError("nope"))
    oa = FakeOpenAlex(error=NotFoundError("nope"))
    out = await _enricher(s2, oa).enrich(_card(doi="10.1/x"))
    assert out == {}
