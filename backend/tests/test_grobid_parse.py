from __future__ import annotations

from pathlib import Path

import pytest
from lattice.ingestion.grobid_client import parse_tei

FIXTURE = Path(__file__).parent / "fixtures" / "sample.tei.xml"

pytest.importorskip("lxml")


def test_parse_tei_extracts_metadata() -> None:
    doc = parse_tei(FIXTURE.read_bytes())
    assert doc.title == "Deep Learning for Copper Price Forecasting"
    assert doc.authors == ["Jane Doe", "Carlos Ng"]
    assert doc.year == 2024
    assert doc.doi == "10.1234/abcd.2024"
    assert doc.venue == "Journal of Commodity Forecasting"
    assert doc.abstract and "LSTM" in doc.abstract


def test_parse_tei_extracts_sections() -> None:
    doc = parse_tei(FIXTURE.read_bytes())
    titles = [s.title for s in doc.sections]
    assert titles == ["Introduction", "Methodology", "Results"]
    method = doc.section_by_keywords("method")
    assert method is not None
    assert "attention" in method.text.lower()


def test_parse_tei_extracts_references() -> None:
    doc = parse_tei(FIXTURE.read_bytes())
    assert len(doc.references) == 2
    by_doi = {r.doi for r in doc.references if r.doi}
    assert "10.5555/var.2019" in by_doi
    arxiv_refs = [r for r in doc.references if r.arxiv_id]
    assert arxiv_refs[0].arxiv_id == "2007.12345"  # normalized (prefix stripped)


def test_full_text_includes_abstract_and_sections() -> None:
    doc = parse_tei(FIXTURE.read_bytes())
    ft = doc.full_text()
    assert "# Abstract" in ft
    assert "# Methodology" in ft
