"""SPECTER2 paper-level embeddings.

Strategy (cost-aware): prefer Semantic Scholar's precomputed ``specter_v2`` vector
when the paper is indexed by S2 (zero local compute). Fall back to a local backend
otherwise. SPECTER2 takes title + abstract as input.
"""

from __future__ import annotations

from dataclasses import dataclass

from lattice.core.logging import get_logger
from lattice.embeddings.base import HashingEmbedder, TextEmbedder, l2_normalize

log = get_logger("embeddings.specter2")


@dataclass
class PaperEmbedding:
    vector: list[float]
    source: str  # "s2_precomputed" | "local"


def specter_input(title: str, abstract: str | None) -> str:
    """SPECTER2 input format: title and abstract joined by the separator token."""
    return f"{title}{' [SEP] ' + abstract if abstract else ''}"


class Specter2Embedder:
    def __init__(self, local_backend: TextEmbedder | None = None, dim: int = 768) -> None:
        # Default to the offline hashing fallback so the system always works.
        self._local = local_backend or HashingEmbedder(dim=dim)
        self._dim = dim

    def from_precomputed(self, vector: list[float] | None) -> PaperEmbedding | None:
        if vector:
            return PaperEmbedding(vector=l2_normalize(vector), source="s2_precomputed")
        return None

    def embed(
        self, title: str, abstract: str | None, *, precomputed: list[float] | None = None
    ) -> PaperEmbedding:
        pre = self.from_precomputed(precomputed)
        if pre is not None:
            return pre
        vec = self._local.embed([specter_input(title, abstract)])[0]
        return PaperEmbedding(vector=l2_normalize(vec), source="local")
