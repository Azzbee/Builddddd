from __future__ import annotations

import pytest
from lattice.api.deps import build_container
from lattice.config import Settings
from lattice.demo import DEMO_CORPUS, load_demo

pytest.importorskip("lxml")


async def test_demo_loads_full_corpus_offline() -> None:
    settings = Settings(demo_mode=True)
    container = build_container(settings)
    n = await load_demo(container)
    assert n == len(DEMO_CORPUS)

    cards = await container.cards.all_cards()
    assert len(cards) == len(DEMO_CORPUS)  # supersede, never delete: both versions stored

    # The wavelet preprint was superseded by its published version during load,
    # so the default graph view shows one fewer node than the corpus has cards.
    superseded = await container.cards.superseded_map()
    assert len(superseded) == 1
    snapshot = await container.ingestion.graph_snapshot()
    assert len(snapshot.nodes) == len(DEMO_CORPUS) - 1
    assert set(superseded) & {node.id for node in snapshot.nodes} == set()
    assert len(snapshot.edges) > 0

    # Contradiction analysis found the transformer dispute and convergent claims.
    relations = await container.ingestion.get_claim_relations()
    kinds = {str(e.relation) for e in relations}
    assert "CONTRADICTS" in kinds


async def test_demo_quadrants_and_momentum_populate() -> None:
    container = build_container(Settings(demo_mode=True))
    await load_demo(container)
    quad = await container.ingestion.epistemic_quadrants()
    assert quad["known_unknowns"]  # open problems clustered from future_work
    digest = await container.ingestion.generate_digest()
    assert "# Lattice digest" in digest["markdown"]  # type: ignore[index]
