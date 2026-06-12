"""Content hashing and deterministic identifiers for idempotency."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid

# Stable namespace so the same logical inputs always map to the same UUIDv5.
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def content_hash(data: bytes) -> str:
    """SHA-256 of raw bytes. Used to dedupe identical PDFs byte-for-byte."""
    return hashlib.sha256(data).hexdigest()


def stable_id(*parts: str) -> str:
    """Deterministic UUIDv5 from string parts. Same inputs -> same id, always."""
    key = "::".join(p.strip().lower() for p in parts if p)
    return str(uuid.uuid5(_NS, key))


_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Shared by dedup (title matching) and entity resolution (method/dataset names)
    so the two stay consistent.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _NONWORD.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def normalize_doi(doi: str | None) -> str | None:
    """Normalize a DOI to its bare ``10.xxxx/...`` lowercase form."""
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi or None


def normalize_arxiv(arxiv_id: str | None) -> str | None:
    """Normalize an arXiv id, stripping URL/prefix and version suffix."""
    if not arxiv_id:
        return None
    aid = arxiv_id.strip().lower()
    aid = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", aid)
    aid = re.sub(r"^arxiv:\s*", "", aid)
    aid = re.sub(r"\.pdf$", "", aid)
    aid = re.sub(r"v\d+$", "", aid)  # drop version so v1/v2 dedupe together
    return aid or None
