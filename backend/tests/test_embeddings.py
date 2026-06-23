from __future__ import annotations

import math

import pytest
from lattice.embeddings.base import HashingEmbedder, l2_normalize, make_text_embedder
from lattice.embeddings.chunks import AspectEmbedder, ChunkEmbedder, aspect_texts
from lattice.embeddings.specter2 import Specter2Embedder, specter_input
from lattice.extraction.schemas import Methodology, PaperCard, Result
from lattice.ingestion.chunker import Chunk


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_make_text_embedder_hashing_when_not_local() -> None:
    emb = make_text_embedder("BAAI/bge-m3", 1024, prefer_local=False)
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == 1024


def test_make_text_embedder_empty_input() -> None:
    assert make_text_embedder("x", 64, prefer_local=False).embed([]) == []


def test_container_uses_hashing_offline() -> None:
    # In dev/demo/test the container must resolve to the hashing backend so the
    # suite stays offline (no torch/model download).
    from lattice.api.deps import build_container
    from lattice.config import Settings

    c = build_container(Settings(environment="dev"))
    assert type(c.chunk_embedder._backend).__name__ == "HashingEmbedder"


def test_hashing_embedder_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["copper price forecasting with lstm"])[0]
    b = emb.embed(["copper price forecasting with lstm"])[0]
    assert a == b
    assert _norm(a) == pytest.approx(1.0)
    assert emb.dim == 64


def test_hashing_embedder_similar_texts_closer() -> None:
    emb = HashingEmbedder(dim=256)
    base = emb.embed(["lstm model for copper price forecasting"])[0]
    similar = emb.embed(["lstm model for copper price prediction"])[0]
    different = emb.embed(["reinforcement learning for robot locomotion"])[0]

    def dot(x: list[float], y: list[float]) -> float:
        return sum(a * b for a, b in zip(x, y, strict=True))

    assert dot(base, similar) > dot(base, different)


def test_l2_normalize_zero_vector_safe() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_specter_input_format() -> None:
    assert specter_input("Title", "Abstract") == "Title [SEP] Abstract"
    assert specter_input("Title", None) == "Title"


def test_specter2_prefers_precomputed() -> None:
    emb = Specter2Embedder(dim=8)
    pre = emb.embed("t", "a", precomputed=[3.0, 4.0] + [0.0] * 6)
    assert pre.source == "s2_precomputed"
    assert _norm(pre.vector) == pytest.approx(1.0)


def test_specter2_local_fallback() -> None:
    emb = Specter2Embedder(dim=32)
    res = emb.embed("Copper forecasting", "We use LSTM models")
    assert res.source == "local"
    assert len(res.vector) == 32


def test_chunk_embedder_returns_vector_per_chunk() -> None:
    chunks = [
        Chunk(chunk_id=f"c{i}", paper_id="p", section_id="s", section_title="S", ordinal=i,
              text=f"text {i}", char_start=0, char_end=6)
        for i in range(5)
    ]
    embedder = ChunkEmbedder(dim=32, batch_size=2)
    vecs = embedder.embed_chunks(chunks)
    assert set(vecs) == {c.chunk_id for c in chunks}
    assert all(len(v) == 32 for v in vecs.values())


def test_aspect_embedder() -> None:
    card = PaperCard(
        paper_id="p",
        title="t",
        problem_statement="Forecasting copper is hard.",
        methodology=Methodology(approach_summary="LSTM with attention"),
        key_results=[Result(claim="beats ARIMA", evidence_location="T1")],
    )
    assert set(aspect_texts(card)) == {"problem", "methodology", "results"}
    vecs = AspectEmbedder(dim=16).embed_card(card)
    assert set(vecs) == {"problem", "methodology", "results"}
    assert all(len(v) == 16 for v in vecs.values())


def test_aspect_empty_field_is_zero_vector() -> None:
    card = PaperCard(
        paper_id="p", title="t",
        problem_statement="Something",
        methodology=Methodology(approach_summary="approach"),
        key_results=[],  # empty results -> zero vector
    )
    vecs = AspectEmbedder(dim=16).embed_card(card)
    assert vecs["results"] == [0.0] * 16
    assert any(x != 0.0 for x in vecs["problem"])
