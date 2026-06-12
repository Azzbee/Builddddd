"""PDF sanity checks that map to actionable, resumable failure states.

These run before the expensive parse so corrupted/scanned/paywalled inputs fail
fast into the job state machine with a precise error code. The text-extraction
step is injectable so the heuristics are testable without pypdf.
"""

from __future__ import annotations

from collections.abc import Callable

from lattice.core.errors import CorruptedPdfError, PaywalledStubError, ScannedPdfError

_PDF_MAGIC = b"%PDF-"
_PAYWALL_MARKERS = (
    "access this article",
    "purchase pdf",
    "sign in to read",
    "institutional login",
    "subscribe to access",
    "buy this article",
    "rent this article",
    "get access",
    "403 forbidden",
)
#: Below this many characters of extractable text we treat the PDF as scanned.
MIN_TEXT_CHARS = 500
#: A "real" paper is rarely under this byte size; smaller is suspicious.
MIN_PDF_BYTES = 3_000


def is_pdf(data: bytes) -> bool:
    """True if the bytes start with a PDF magic header (allowing leading junk)."""
    return _PDF_MAGIC in data[:1024]


def looks_like_paywall_stub(text: str, size_bytes: int) -> bool:
    lowered = text.lower()
    marker_hit = any(m in lowered for m in _PAYWALL_MARKERS)
    # Paywall landing pages are short and contain access-gate language.
    return marker_hit and (len(text) < MIN_TEXT_CHARS or size_bytes < 50_000)


def _default_extract_text(data: bytes) -> str:  # pragma: no cover - needs pypdf
    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise CorruptedPdfError(f"could not read PDF: {exc}") from exc


def classify_pdf(
    data: bytes,
    extract_text: Callable[[bytes], str] | None = None,
) -> str:
    """Validate a PDF and return its extracted text, or raise a typed error.

    Raises :class:`CorruptedPdfError`, :class:`ScannedPdfError`, or
    :class:`PaywalledStubError`. Returns the extracted text on success.
    """
    if not data:
        raise CorruptedPdfError("empty file")
    if len(data) < MIN_PDF_BYTES:
        raise CorruptedPdfError(f"file too small ({len(data)} bytes)")
    if not is_pdf(data):
        raise CorruptedPdfError("missing PDF header")

    extractor = extract_text or _default_extract_text
    text = extractor(data)

    if looks_like_paywall_stub(text, len(data)):
        raise PaywalledStubError("content looks like a paywall landing page")
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ScannedPdfError(
            f"only {len(text.strip())} chars of text; likely scanned (needs OCR/vision)"
        )
    return text
