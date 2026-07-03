from __future__ import annotations

from lattice.export.obsidian import card_to_note
from lattice.extraction.schemas import Author, DatasetRef, Methodology, PaperCard, Result
from lattice.rag.related_work import (
    bibtex_key,
    build_related_work,
    corpus_to_bibtex,
    to_bibtex,
)


def _card(pid: str, title: str, methods: list[str], domain: str) -> PaperCard:
    return PaperCard(
        paper_id=pid,
        title=title,
        authors=[Author(name="Jane Doe"), Author(name="Carlos Ng")],
        year=2024,
        venue="Journal of Forecasting",
        doi=f"10.1/{pid}",
        methodology=Methodology(approach_summary="approach", techniques=methods),
        methods_taxonomy=methods,
        datasets=[DatasetRef(name="LME Copper")],
        key_results=[Result(claim="beats ARIMA", evidence_location="T1")],
        domains=[domain],
        limitations=["single commodity"],
        contributions=["new model"],
    )


def test_bibtex_key_stable() -> None:
    c = _card("p1", "Deep Learning for Copper Forecasting", ["LSTM"], "commodity markets")
    assert bibtex_key(c) == "doe2024deep"  # firstauthor surname + year + first non-stopword


def test_to_bibtex_fields() -> None:
    c = _card("p1", "Copper Forecasting", ["LSTM"], "commodity markets")
    bib = to_bibtex(c)
    assert "@article{doe2024copper" in bib
    assert "author = {Jane Doe and Carlos Ng}" in bib
    assert "doi = {10.1/p1}" in bib
    assert "year = {2024}" in bib


def test_corpus_to_bibtex_dedupes() -> None:
    a = _card("p1", "Copper Forecasting", ["LSTM"], "x")
    b = _card("p1", "Copper Forecasting", ["LSTM"], "x")  # same paper_id -> one entry
    bib = corpus_to_bibtex([a, b])
    assert bib.count("@article") == 1


def test_corpus_to_bibtex_disambiguates_distinct_key_collisions() -> None:
    # Two DISTINCT papers whose base key collides (same surname+year+first title
    # word) must both be emitted with unique keys, not silently dropped.
    a = _card("p1", "Copper Forecasting with LSTM", ["LSTM"], "x")
    b = _card("p2", "Copper Forecasting with GRU", ["GRU"], "x")
    assert bibtex_key(a) == bibtex_key(b) == "doe2024copper"  # they collide
    bib = corpus_to_bibtex([a, b])
    assert bib.count("@article") == 2  # both kept
    assert "@article{doe2024copper," in bib
    assert "@article{doe2024coppera," in bib  # second disambiguated


def test_assign_unique_keys_handles_many_collisions_and_base_clash() -> None:
    from lattice.rag.related_work import assign_unique_keys

    # 30 distinct papers that all collide on the same base key: suffixes must stay
    # valid (b..z, then aa, ab, ...) past the 26-letter boundary, never emitting a
    # non-letter like '{'.
    cards = [_card(f"p{i}", "Copper Forecasting with LSTM", ["LSTM"], "x") for i in range(30)]
    keys = assign_unique_keys(cards)
    assert len(set(keys.values())) == 30  # all unique
    for k in keys.values():
        assert k == "doe2024copper" or k[len("doe2024copper") :].isalpha()
    assert "doe2024copperaa" in keys.values()  # 27th distinct key (i=26 -> 'aa')

    # A paper whose base key literally equals another paper's suffixed key must not
    # collide: assignment stays globally unique.
    a = _card("p1", "Copper Forecasting", ["LSTM"], "x")  # -> doe2024copper
    b = _card("p2", "Copper Forecasting", ["LSTM"], "x")  # -> doe2024copper (collides -> ...b)
    clash = _card("p3", "Copperb Forecasting", ["LSTM"], "x")  # base key -> doe2024copperb
    keys2 = assign_unique_keys([a, b, clash])
    assert len(set(keys2.values())) == 3


def test_related_work_cite_markers_match_unique_keys() -> None:
    # The [@key] markers in the summary, the papers-list keys, and the BibTeX must
    # all agree on the disambiguated key for a collision.
    groups = {
        "0": [
            _card("p1", "Copper Forecasting with LSTM", ["LSTM"], "commodity markets"),
            _card("p2", "Copper Forecasting with GRU", ["GRU"], "commodity markets"),
        ],
    }
    draft = build_related_work(groups)
    cluster = draft.clusters[0]
    marker_keys = {p["key"] for p in cluster.to_json()["papers"]}  # type: ignore[index]
    assert marker_keys == {"doe2024copper", "doe2024coppera"}
    for k in marker_keys:
        assert f"[@{k}]" in cluster.summary
        assert f"@article{{{k}," in draft.bibtex()


def test_build_related_work_groups_and_hedges() -> None:
    groups = {
        "0": [
            _card("p1", "LSTM Copper", ["LSTM"], "commodity markets"),
            _card("p2", "GRU Copper", ["GRU"], "commodity markets"),
        ],
        "1": [_card("p3", "Lonely VAR", ["VAR"], "econometrics")],  # singleton -> hedged
    }
    draft = build_related_work(groups)
    assert len(draft.clusters) == 2
    # Larger cluster first.
    assert len(draft.clusters[0].cards) == 2
    assert draft.clusters[0].label == "Commodity Markets"
    singleton = next(cl for cl in draft.clusters if len(cl.cards) == 1)
    assert singleton.hedged
    assert "preliminary" in singleton.summary
    md = draft.markdown()
    assert "# Related work" in md and "[@doe2024lstm]" in md
    assert draft.bibtex().count("@article") == 3


def test_card_to_note_obsidian_wikilinks() -> None:
    c = _card("p1", "Copper Forecasting", ["LSTM"], "commodity markets")
    note = card_to_note(c, neighbors=[("GRU Copper", 0.82), ("VAR Baseline", 0.41)])
    assert note.startswith("---")
    assert 'title: "Copper Forecasting"' in note
    assert "## Related" in note
    assert "[[GRU Copper]] (weight 0.82)" in note
    # Sorted by weight descending.
    assert note.index("GRU Copper") < note.index("VAR Baseline")
