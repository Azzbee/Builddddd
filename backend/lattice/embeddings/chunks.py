"""Chunk and aspect embeddings (embedding strategy layer 2).

Chunk embeddings power hybrid retrieval. Aspect embeddings embed PaperCard fields
separately (problem statement, methodology summary, concatenated key-result
claims) so the similarity function gets per-aspect signals (thematic,
methodological, empirical) that sidestep SPECTER2's domain bias because the input
is normalized, LLM-distilled text rather than raw domain prose.
"""

from __future__ import annotations

from lattice.embeddings.base import HashingEmbedder, TextEmbedder, l2_normalize
from lattice.extraction.schemas import PaperCard
from lattice.ingestion.chunker import Chunk


class ChunkEmbedder:
    def __init__(self, backend: TextEmbedder | None = None, dim: int = 1024, batch_size: int = 16):
        self._backend = backend or HashingEmbedder(dim=dim)
        self._batch = batch_size

    @property
    def dim(self) -> int:
        return self._backend.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            out.extend(l2_normalize(v) for v in self._backend.embed(batch))
        return out

    def embed_chunks(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        """Return ``{chunk_id: vector}``."""
        vecs = self.embed_texts([c.text for c in chunks])
        return {c.chunk_id: v for c, v in zip(chunks, vecs, strict=True)}


def aspect_texts(card: PaperCard) -> dict[str, str]:
    """Build the three aspect strings from a PaperCard."""
    claims = " ".join(r.claim for r in card.key_results)
    return {
        "problem": card.problem_statement,
        "methodology": card.methodology.approach_summary,
        "results": claims,
    }


class AspectEmbedder:
    def __init__(self, backend: TextEmbedder | None = None, dim: int = 1024):
        self._backend = backend or HashingEmbedder(dim=dim)

    @property
    def dim(self) -> int:
        return self._backend.dim

    def embed_card(self, card: PaperCard) -> dict[str, list[float]]:
        aspects = aspect_texts(card)
        keys = list(aspects)
        # Embed empty aspects as zero vectors (similarity treats them as missing).
        texts = [aspects[k] or "" for k in keys]
        vecs = self._backend.embed(texts)
        out: dict[str, list[float]] = {}
        for k, text, v in zip(keys, texts, vecs, strict=True):
            out[k] = l2_normalize(v) if text.strip() else [0.0] * self._backend.dim
        return out
