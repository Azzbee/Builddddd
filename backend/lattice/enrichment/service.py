"""Composite enricher wiring Semantic Scholar + OpenAlex into the pipeline.

Implements the ingestion ``Enricher`` protocol: given a PaperCard, return a dict of
enrichment signals consumed by the embedding/linking stages:

* ``specter_embedding`` - free precomputed SPECTER2 vector (S2), avoiding local inference
* ``reference_ids`` - cited paper ids for bibliographic-coupling (S_cit)
* ``s2_paper_id`` / ``citation_count`` / ``concepts`` - metadata + ADDRESSES concepts

Best-effort by construction: missing records or rate limits degrade to text-only
similarity (the service swallows exceptions, and this returns what it could get).
"""

from __future__ import annotations

import contextlib

from lattice.config import EnrichmentSettings
from lattice.core.errors import EnrichmentError
from lattice.core.logging import get_logger
from lattice.enrichment.openalex import OpenAlexClient
from lattice.enrichment.semantic_scholar import SemanticScholarClient
from lattice.extraction.schemas import PaperCard

log = get_logger("enrichment.service")


class CompositeEnricher:
    def __init__(
        self,
        settings: EnrichmentSettings,
        s2: SemanticScholarClient | None = None,
        openalex: OpenAlexClient | None = None,
    ) -> None:
        self._s2 = s2 or SemanticScholarClient(settings)
        self._openalex = openalex or OpenAlexClient(settings)

    async def enrich(self, card: PaperCard) -> dict[str, object]:
        out: dict[str, object] = {}
        has_id = bool(card.doi or card.arxiv_id or card.s2_paper_id)

        if has_id:
            with contextlib.suppress(EnrichmentError, ValueError):
                paper = await self._s2.get_paper(
                    doi=card.doi, arxiv_id=card.arxiv_id, s2_id=card.s2_paper_id
                )
                if paper.s2_paper_id:
                    out["s2_paper_id"] = paper.s2_paper_id
                if paper.reference_ids:
                    out["reference_ids"] = paper.reference_ids
                if paper.citation_count is not None:
                    out["citation_count"] = paper.citation_count
                if paper.specter_embedding:
                    out["specter_embedding"] = paper.specter_embedding

        # Fall back to OpenAlex for references (and concepts) when S2 lacked them.
        if card.doi and not out.get("reference_ids"):
            with contextlib.suppress(EnrichmentError):
                work = await self._openalex.get_work(doi=card.doi)
                out.setdefault("reference_ids", work.reference_ids)
                if work.openalex_id:
                    # The paper's own OpenAlex id, in the same URL form OpenAlex
                    # reference lists use. Without it, direct-citation detection
                    # can never match an OpenAlex-sourced reference to this paper.
                    out["openalex_id"] = work.openalex_id
                if work.concepts:
                    out["concepts"] = work.concepts

        reference_ids = out.get("reference_ids")
        reference_count = len(reference_ids) if isinstance(reference_ids, list) else 0
        log.info(
            "enrich.result",
            paper_id=card.paper_id,
            has_specter="specter_embedding" in out,
            refs=reference_count,
        )
        return out
