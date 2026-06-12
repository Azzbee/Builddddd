"""Entity resolution for Method / Dataset / Concept nodes.

When a new paper introduces an entity that fuzzy-matches an existing node, we MERGE
into the existing node instead of creating a near-duplicate. The cascade is:

1. exact normalized-name match -> merge
2. high fuzzy score (rapidfuzz) -> merge
3. ambiguous band -> break the tie with embedding cosine similarity if available
4. still ambiguous -> optional LLM tiebreaker
5. otherwise -> new canonical entity

Keeping this pure (entities held in an in-memory registry, embedder/LLM injected)
makes the cascade fully testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from lattice.core.hashing import normalize_text, stable_id
from lattice.graph.similarity import cosine

# Thresholds on a 0-1 scale.
AUTO_MATCH = 0.93
AMBIGUOUS_FLOOR = 0.80
EMBED_MATCH = 0.86

TiebreakFn = Callable[[str, str], bool]


@dataclass
class CanonicalEntity:
    key: str
    name: str
    embedding: list[float] | None = None
    aliases: set[str] = field(default_factory=set)


@dataclass
class Resolution:
    matched: bool
    key: str
    name: str
    score: float
    method: str  # "exact" | "fuzzy" | "embedding" | "llm" | "new"


class EntityResolver:
    """Resolves entity names to canonical keys within one entity type + workspace."""

    def __init__(
        self,
        entity_type: str,
        workspace_id: str = "default",
        tiebreaker: TiebreakFn | None = None,
    ) -> None:
        self._type = entity_type
        self._workspace = workspace_id
        self._tiebreaker = tiebreaker
        self._by_norm: dict[str, CanonicalEntity] = {}
        self._entities: list[CanonicalEntity] = []

    def make_key(self, name: str) -> str:
        return stable_id(self._workspace, self._type, normalize_text(name))

    def register(self, name: str, embedding: list[float] | None = None) -> CanonicalEntity:
        norm = normalize_text(name)
        existing = self._by_norm.get(norm)
        if existing:
            if embedding and existing.embedding is None:
                existing.embedding = embedding
            return existing
        entity = CanonicalEntity(key=self.make_key(name), name=name, embedding=embedding)
        self._by_norm[norm] = entity
        self._entities.append(entity)
        return entity

    def resolve(self, name: str, embedding: list[float] | None = None) -> Resolution:
        norm = normalize_text(name)
        if not norm:
            key = self.make_key(name)
            return Resolution(False, key, name, 0.0, "new")

        exact = self._by_norm.get(norm)
        if exact:
            return Resolution(True, exact.key, exact.name, 1.0, "exact")

        best: CanonicalEntity | None = None
        best_score = 0.0
        for ent in self._entities:
            score = fuzz.token_sort_ratio(norm, normalize_text(ent.name)) / 100.0
            if score > best_score:
                best_score, best = score, ent

        if best is not None and best_score >= AUTO_MATCH:
            best.aliases.add(name)
            return Resolution(True, best.key, best.name, best_score, "fuzzy")

        # Ambiguous band: try embedding similarity, then LLM tiebreak.
        if best is not None and best_score >= AMBIGUOUS_FLOOR:
            if embedding is not None and best.embedding is not None:
                import numpy as np

                sim = cosine(np.asarray(embedding), np.asarray(best.embedding))
                if sim is not None and sim >= EMBED_MATCH:
                    best.aliases.add(name)
                    return Resolution(True, best.key, best.name, sim, "embedding")
            if self._tiebreaker is not None and self._tiebreaker(name, best.name):
                best.aliases.add(name)
                return Resolution(True, best.key, best.name, best_score, "llm")

        # New entity: register and return.
        entity = self.register(name, embedding)
        return Resolution(False, entity.key, entity.name, best_score, "new")
