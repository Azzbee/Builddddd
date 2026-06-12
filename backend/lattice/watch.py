"""arXiv watch filtering: score fetched candidates by similarity to the corpus.

The watcher fetches recent papers (enrichment.arxiv_watcher); this module scores
each against the existing corpus embeddings and keeps those above a floor as an
approval queue. Nothing is auto-ingested: matches go to a queue the user approves,
per the PRD. Pure over an injected embedder (offline HashingEmbedder by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lattice.embeddings.base import HashingEmbedder, TextEmbedder
from lattice.enrichment.models import ArxivResult
from lattice.graph.similarity import cosine


@dataclass
class ScoredCandidate:
    candidate: ArxivResult
    similarity: float
    nearest_paper_id: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "arxiv_id": self.candidate.arxiv_id,
            "title": self.candidate.title,
            "similarity": round(self.similarity, 4),
            "nearest_paper_id": self.nearest_paper_id,
            "pdf_url": self.candidate.pdf_url,
        }


def score_candidates(
    candidates: list[ArxivResult],
    corpus_vectors: list[tuple[str, list[float]]],
    *,
    embedder: TextEmbedder | None = None,
    floor: float = 0.45,
) -> list[ScoredCandidate]:
    """Return candidates whose max similarity to the corpus is >= floor, ranked.

    Already-present arXiv ids (by exact match in the corpus) are not re-surfaced.
    """
    embedder = embedder or HashingEmbedder(dim=768)
    if not candidates or not corpus_vectors:
        return []

    corpus_ids = [pid for pid, _ in corpus_vectors]
    corpus_mat = [np.asarray(v, dtype=float) for _, v in corpus_vectors]

    texts = [f"{c.title}. {c.summary}" for c in candidates]
    cand_vecs = embedder.embed(texts)

    scored: list[ScoredCandidate] = []
    for cand, vec in zip(candidates, cand_vecs, strict=True):
        cv = np.asarray(vec, dtype=float)
        best_sim = 0.0
        best_id: str | None = None
        for pid, cm in zip(corpus_ids, corpus_mat, strict=True):
            sim = cosine(cv, cm)
            if sim is not None and sim > best_sim:
                best_sim, best_id = sim, pid
        if best_sim >= floor:
            scored.append(ScoredCandidate(cand, best_sim, best_id))

    scored.sort(key=lambda s: s.similarity, reverse=True)
    return scored
