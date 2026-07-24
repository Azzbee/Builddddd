from __future__ import annotations

from lattice.landscape.reading_queue import (
    ReadingCandidate,
    expected_information_gain,
    rank_reading_queue,
)


def test_novelty_and_gain_components() -> None:
    c = ReadingCandidate(
        paper_id="p1",
        title="Novel",
        methods={"diffusion", "lstm"},
        neighbors=[(0.8, 0.9), (0.5, 0.4)],
    )
    s = expected_information_gain(c, corpus_methods={"lstm"})
    assert s.method_novelty == 0.5  # diffusion is novel, lstm is known
    assert "diffusion" in s.novel_methods
    assert s.neighborhood_value > 0
    assert s.gain > 0


def test_more_central_and_novel_ranks_higher() -> None:
    central_novel = ReadingCandidate("a", "A", {"diffusion"}, [(0.9, 0.9)])
    peripheral_known = ReadingCandidate("b", "B", {"lstm"}, [(0.2, 0.1)])
    ranked = rank_reading_queue([peripheral_known, central_novel], corpus_methods={"lstm"})
    assert ranked[0].paper_id == "a"


def test_no_neighbors_zero_neighborhood() -> None:
    c = ReadingCandidate("p", "P", {"x"}, [])
    s = expected_information_gain(c, set())
    assert s.neighborhood_value == 0.0
    assert s.gain == 0.0


def test_rank_respects_limit() -> None:
    candidates = [ReadingCandidate(f"p{i}", f"P{i}", {"m"}, [(0.5, 0.5)]) for i in range(30)]
    assert len(rank_reading_queue(candidates, set(), limit=5)) == 5
