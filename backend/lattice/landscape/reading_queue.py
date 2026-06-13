"""Reading-queue ranking: which unread paper has the highest expected information
gain, given your graph.

Expected information gain blends two signals (both visible, like the edge weights):

* neighborhood value - how strongly the paper connects to what you already have,
  weighted by the importance of its neighbors. A paper wired into a central cluster
  teaches you more about that cluster.
* method novelty - how many of its methods are new relative to your corpus. A paper
  that only repeats known methods has lower marginal value.

    gain = neighborhood_value * (0.5 + method_novelty)

Pure functions over simple inputs so it is fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from lattice.core.hashing import normalize_text


@dataclass
class ReadingCandidate:
    paper_id: str
    title: str
    methods: set[str]
    #: (edge_weight, neighbor_centrality) for each RELATED neighbor already read.
    neighbors: list[tuple[float, float]]


@dataclass
class ScoredReading:
    paper_id: str
    title: str
    gain: float
    neighborhood_value: float
    method_novelty: float
    novel_methods: list[str]

    def to_json(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "gain": round(self.gain, 4),
            "neighborhood_value": round(self.neighborhood_value, 4),
            "method_novelty": round(self.method_novelty, 4),
            "novel_methods": self.novel_methods,
        }


def _novelty(methods: set[str], corpus_methods: set[str]) -> tuple[float, list[str]]:
    norm = {normalize_text(m) for m in methods if m.strip()}
    corpus = {normalize_text(m) for m in corpus_methods if m.strip()}
    if not norm:
        return 0.0, []
    novel = norm - corpus
    return len(novel) / len(norm), sorted(novel)


def expected_information_gain(c: ReadingCandidate, corpus_methods: set[str]) -> ScoredReading:
    neighborhood = sum(w * (0.5 + nc) for w, nc in c.neighbors)
    novelty, novel = _novelty(c.methods, corpus_methods)
    gain = neighborhood * (0.5 + novelty)
    return ScoredReading(
        paper_id=c.paper_id,
        title=c.title,
        gain=gain,
        neighborhood_value=neighborhood,
        method_novelty=novelty,
        novel_methods=novel,
    )


def rank_reading_queue(
    candidates: list[ReadingCandidate], corpus_methods: set[str], *, limit: int = 20
) -> list[ScoredReading]:
    scored = [expected_information_gain(c, corpus_methods) for c in candidates]
    scored.sort(key=lambda s: s.gain, reverse=True)
    return scored[:limit]
