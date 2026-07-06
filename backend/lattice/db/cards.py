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

    Everything here survives a process restart: the SPECTER paper vector, the
    normalized method/dataset sets, and the per-aspect embeddings (problem /
    methodology / results). Rehydrated papers therefore link with full similarity
    fidelity (sem + methodology-section + method-tags + dataset), and the epistemic
    quadrants keep their aspect vectors across restarts.
    """

    paper_id: str
    specter: list[float] | None
    methods: set[str]
    datasets: set[str]
    aspects: dict[str, list[float]] | None = None


class CorpusStore(Protocol):
    """Full card store surface used by the ingestion service (in-memory or Postgres)."""

    async def get_card(self, paper_id: str) -> dict[str, Any] | None: ...
    async def put_card(
        self,
        card: PaperCard,
        specter: list[float] | None = None,
        aspects: dict[str, list[float]] | None = None,
    ) -> None: ...
    async def get(self, paper_id: str) -> PaperCard | None: ...
    async def all_cards(self) -> list[PaperCard]: ...
    async def corpus_index(self) -> CorpusIndex: ...
    async def load_features(self) -> list[StoredFeatures]: ...
    async def mark_superseded(self, paper_id: str, superseded_by: str) -> None: ...
    async def superseded_ids(self) -> set[str]: ...


class JobStore(Protocol):
    async def save(self, job: IngestJob) -> None: ...
    async def get(self, job_id: str) -> IngestJob | None: ...
    async def all_jobs(self) -> list[IngestJob]: ...


class InMemoryCardStore:
    def __init__(self) -> None:
        self._cards: dict[str, PaperCard] = {}
        self._specters: dict[str, list[float] | None] = {}
        self._aspects: dict[str, dict[str, list[float]]] = {}
        self._superseded: dict[str, str] = {}  # paper_id -> superseding paper_id

    async def put_card(
        self,
        card: PaperCard,
        specter: list[float] | None = None,
        aspects: dict[str, list[float]] | None = None,
    ) -> None:
        self._cards[card.paper_id] = card
        if specter is not None:
            self._specters[card.paper_id] = specter
        if aspects is not None:
            self._aspects[card.paper_id] = aspects

    async def get_card(self, paper_id: str) -> dict[str, Any] | None:
        card = self._cards.get(paper_id)
        return card.model_dump(mode="json") if card else None

    async def get(self, paper_id: str) -> PaperCard | None:
        return self._cards.get(paper_id)

    async def all_cards(self) -> list[PaperCard]:
        return list(self._cards.values())

    async def load_features(self) -> list[StoredFeatures]:
        # Superseded papers are excluded: they must not re-enter the linking
        # candidate pool after a restart (their edges are invalidated, and new
        # papers should link to the superseding version instead).
        return [
            StoredFeatures(
                paper_id=c.paper_id,
                specter=self._specters.get(c.paper_id),
                methods=c.normalized_methods,
                datasets=c.normalized_datasets,
                aspects=self._aspects.get(c.paper_id),
            )
            for c in self._cards.values()
            if c.paper_id not in self._superseded
        ]

    async def mark_superseded(self, paper_id: str, superseded_by: str) -> None:
        self._superseded[paper_id] = superseded_by

    async def superseded_ids(self) -> set[str]:
        return set(self._superseded)

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
