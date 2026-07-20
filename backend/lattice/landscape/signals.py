"""Global and demand signals for the gap matrix.

The corpus alone tells you what you have read, not what the world is researching.
These providers add:

* ``DemandScorer`` - local signal: how strongly the corpus's own future-work and
  limitations point at a cell (embedding similarity to a cell description). Always
  available, no network.
* ``fetch_global_counts`` - global signal via OpenAlex: how many papers exist
  worldwide for a cell. This is what separates a true gap (Empty) from a reading
  gap (Blind spot). Degrades to zero counts when OpenAlex is unreachable.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

import numpy as np

from lattice.embeddings.base import TextEmbedder
from lattice.extraction.schemas import PaperCard
from lattice.graph.similarity import cosine


def cell_description(row: str, col: str) -> str:
    return f"{row} applied to {col}"


@dataclass
class DemandScorer:
    """Scores how much the corpus's future-work/limitations demand each cell."""

    embedder: TextEmbedder
    _paper_vectors: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_cards(cls, embedder: TextEmbedder, cards: list[PaperCard]) -> DemandScorer:
        texts = [" ".join([*card.future_work, *card.limitations]).strip() for card in cards]
        nonempty = [t for t in texts if t]
        vectors = [np.asarray(v, dtype=float) for v in embedder.embed(nonempty)] if nonempty else []
        return cls(embedder=embedder, _paper_vectors=vectors)

    def score(self, row: str, col: str) -> float:
        if not self._paper_vectors:
            return 0.0
        desc = np.asarray(self.embedder.embed([cell_description(row, col)])[0], dtype=float)
        best = 0.0
        for vec in self._paper_vectors:
            sim = cosine(desc, vec)
            if sim is not None:
                best = max(best, sim)
        return max(0.0, min(1.0, best))


async def fetch_global_counts(
    openalex: object,  # OpenAlexClient (duck-typed to keep landscape import-light)
    cells: list[tuple[str, str]],
    *,
    concurrency: int = 5,
    cap: int = 60,
) -> dict[tuple[str, str], int]:
    """Fetch OpenAlex works counts for empty cells. Graceful: errors -> skipped.

    ``cap`` bounds how many cells we query so a large matrix does not hammer the
    API; the caller should pass the highest-pressure empties first.
    """
    counts: dict[tuple[str, str], int] = {}
    work = cells[:cap]
    if not work or not hasattr(openalex, "works_count"):
        return counts
    sem = asyncio.Semaphore(concurrency)

    async def one(cell: tuple[str, str]) -> None:
        async with sem:
            with contextlib.suppress(Exception):  # global signal is best-effort
                counts[cell] = await openalex.works_count(f"{cell[0]} {cell[1]}")

    await asyncio.gather(*(one(c) for c in work))
    return counts
