"""Tests for PDF blob storage and page-count metadata."""

from __future__ import annotations

import io

import pytest
from lattice.core.hashing import content_hash
from lattice.db.blobs import BlobMeta, InMemoryBlobStore, pdf_page_count


def _real_pdf(pages: int = 1) -> bytes:
    """A genuinely parseable PDF built with pypdf, so page counts are real."""
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


_MINIMAL_PDF = _real_pdf(1)


async def test_put_get_meta_roundtrip() -> None:
    store = InMemoryBlobStore()
    meta = await store.put("p1", _MINIMAL_PDF)
    assert isinstance(meta, BlobMeta)
    assert meta.size == len(_MINIMAL_PDF)
    assert meta.content_hash == content_hash(_MINIMAL_PDF)
    assert await store.get("p1") == _MINIMAL_PDF
    fetched = await store.meta("p1")
    assert fetched is not None and fetched.paper_id == "p1"


async def test_missing_blob_returns_none() -> None:
    store = InMemoryBlobStore()
    assert await store.get("nope") is None
    assert await store.meta("nope") is None


async def test_explicit_content_hash_is_kept() -> None:
    store = InMemoryBlobStore()
    meta = await store.put("p1", b"%PDF-1.7 junk", content_hash="deadbeef")
    assert meta.content_hash == "deadbeef"


async def test_page_count_tolerates_non_pdf_bytes() -> None:
    # Synthetic/non-parseable bytes -> 0 pages, never raises.
    assert pdf_page_count(b"not a pdf") == 0
    meta = await InMemoryBlobStore().put("p", b"%PDF-1.7\n" + b"x" * 100)
    assert meta.pages == 0


async def test_page_count_real_pdf() -> None:
    assert pdf_page_count(_MINIMAL_PDF) == 1


async def test_meta_to_json_marks_available() -> None:
    meta = await InMemoryBlobStore().put("p1", _MINIMAL_PDF)
    payload = meta.to_json()
    assert payload["available"] is True
    assert payload["pages"] == 1
