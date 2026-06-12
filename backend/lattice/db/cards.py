"""Card and job stores. In-memory implementations for offline running/testing."""

from __future__ import annotations

from typing import Any, Protocol

from lattice.extraction.schemas import PaperCard
from lattice.ingestion.dedup import CorpusIndex, PaperIdentity
from lattice.ingestion.models import IngestJob


class CardStore(Protocol):
    async def get_card(self, paper_id: str) -> dict[str, Any] | None: ...
    async def put_card(self, card: PaperCard) -> None: ...


class InMemoryCardStore:
    def __init__(self) -> None:
        self._cards: dict[str, PaperCard] = {}

    async def put_card(self, card: PaperCard) -> None:
        self._cards[card.paper_id] = card

    async def get_card(self, paper_id: str) -> dict[str, Any] | None:
        card = self._cards.get(paper_id)
        return card.model_dump(mode="json") if card else None

    async def get(self, paper_id: str) -> PaperCard | None:
        return self._cards.get(paper_id)

    async def all_cards(self) -> list[PaperCard]:
        return list(self._cards.values())

    async def corpus_index(self) -> CorpusIndex:
        idents = [
            PaperIdentity(
                paper_id=c.paper_id,
                title=c.title,
                authors=[a.name for a in c.authors],
                doi=c.doi,
                arxiv_id=c.arxiv_id,
            )
            for c in self._cards.values()
        ]
        return CorpusIndex.from_identities(idents)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}

    async def save(self, job: IngestJob) -> None:
        self._jobs[job.job_id] = job.model_copy(deep=True)

    async def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    async def all_jobs(self) -> list[IngestJob]:
        return list(self._jobs.values())
