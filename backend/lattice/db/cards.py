"""Card and job stores. In-memory implementations for offline running/testing."""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

from lattice.extraction.schemas import PaperCard
from lattice.ingestion.dedup import CorpusIndex, PaperIdentity
from lattice.ingestion.models import IngestJob


class CardStore(Protocol):
    """Minimal read surface used by the RAG tools."""

    async def get_card(self, paper_id: str) -> dict[str, Any] | None: ...
    async def put_card(self, card: PaperCard) -> None: ...


class StoredFeatures(NamedTuple):
    """The persisted, per-paper feature bundle used to rehydrate incremental linking.

    Everything here survives a process restart. The methodology-section embedding is
    NOT persisted (it is a transient aspect vector), so a rehydrated paper links on
    sem + method-tags + citation + dataset; the similarity function renormalizes over
    the available components, so this degrades gracefully rather than dropping edges.
    """

    paper_id: str
    specter: list[float] | None
    methods: set[str]
    datasets: set[str]


class CorpusStore(Protocol):
    """Full card store surface used by the ingestion service (in-memory or Postgres)."""

    async def get_card(self, paper_id: str) -> dict[str, Any] | None: ...
    async def put_card(self, card: PaperCard, specter: list[float] | None = None) -> None: ...
    async def get(self, paper_id: str) -> PaperCard | None: ...
    async def all_cards(self) -> list[PaperCard]: ...
    async def corpus_index(self) -> CorpusIndex: ...
    async def load_features(self) -> list[StoredFeatures]: ...


class JobStore(Protocol):
    async def save(self, job: IngestJob) -> None: ...
    async def get(self, job_id: str) -> IngestJob | None: ...
    async def all_jobs(self) -> list[IngestJob]: ...


class InMemoryCardStore:
    def __init__(self) -> None:
        self._cards: dict[str, PaperCard] = {}
        self._specters: dict[str, list[float] | None] = {}

    async def put_card(self, card: PaperCard, specter: list[float] | None = None) -> None:
        self._cards[card.paper_id] = card
        if specter is not None:
            self._specters[card.paper_id] = specter

    async def get_card(self, paper_id: str) -> dict[str, Any] | None:
        card = self._cards.get(paper_id)
        return card.model_dump(mode="json") if card else None

    async def get(self, paper_id: str) -> PaperCard | None:
        return self._cards.get(paper_id)

    async def all_cards(self) -> list[PaperCard]:
        return list(self._cards.values())

    async def load_features(self) -> list[StoredFeatures]:
        return [
            StoredFeatures(
                paper_id=c.paper_id,
                specter=self._specters.get(c.paper_id),
                methods=c.normalized_methods,
                datasets=c.normalized_datasets,
            )
            for c in self._cards.values()
        ]

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
