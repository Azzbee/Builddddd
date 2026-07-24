"""Docling client: layout-aware tables and Markdown body text.

Docling is the region specialist. GROBID stays the backbone for structure and
references; Docling handles its known weaknesses (tables, complex layout). The
:func:`reconcile` function (pure, tested) compares GROBID and Docling text for the
same region by normalized token overlap and decides whether to accept or escalate
to the vision fallback, attaching a ``parse_confidence`` to each region.
"""

from __future__ import annotations

from dataclasses import dataclass

from lattice.config import DoclingSettings
from lattice.core.hashing import normalize_text
from lattice.core.logging import get_logger
from lattice.ingestion.models import TableArtifact

log = get_logger("ingestion.docling")


def token_overlap(a: str, b: str) -> float:
    """Normalized token-set overlap (Jaccard) of two text blocks, in [0, 1]."""
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class ReconcileDecision:
    accept: bool
    confidence: float
    needs_vision: bool
    text: str


@dataclass
class DoclingOutput:
    markdown: str
    tables: list[TableArtifact]


def reconcile(grobid_text: str, docling_text: str, threshold: float) -> ReconcileDecision:
    """Decide which text to trust for a region.

    High agreement -> accept (prefer the richer Docling text). Low agreement ->
    flag for the vision fallback to arbitrate, with confidence equal to the
    measured overlap so downstream extraction can hedge.
    """
    overlap = token_overlap(grobid_text, docling_text)
    if overlap >= threshold:
        # Prefer Docling's layout-aware text when it is non-empty.
        text = docling_text.strip() or grobid_text.strip()
        return ReconcileDecision(accept=True, confidence=overlap, needs_vision=False, text=text)
    # Disagreement: keep the longer text provisionally, mark for vision arbitration.
    text = max((grobid_text, docling_text), key=lambda s: len(s.strip()))
    return ReconcileDecision(accept=False, confidence=overlap, needs_vision=True, text=text)


class DoclingClient:
    """Thin wrapper around the Docling converter (optional dependency).

    Only imported when enabled, so the core test suite never needs Docling/torch.
    Returns structured :class:`TableArtifact` objects (cells, not just Markdown) so
    table results can be linked to Result evidence locations.
    """

    def __init__(self, settings: DoclingSettings):
        self._settings = settings

    def available(self) -> bool:
        try:
            import docling  # noqa: F401
        except ImportError:
            return False
        return self._settings.enabled

    def extract(self, pdf_path: str) -> DoclingOutput:  # pragma: no cover
        """Convert once and return both Markdown body text and structured tables."""
        if not self.available():
            log.warning("docling.unavailable", path=pdf_path)
            return DoclingOutput(markdown="", tables=[])
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(pdf_path)
        document = result.document
        tables: list[TableArtifact] = []
        for i, table in enumerate(getattr(document, "tables", [])):
            try:
                frame = table.export_to_dataframe()
                headers = [str(column) for column in frame.columns]
                rows = [[str(value) for value in row] for row in frame.to_numpy().tolist()]
            except Exception:
                headers, rows = [], []
            tables.append(
                TableArtifact(
                    table_id=f"table-{i + 1}",
                    caption=getattr(table, "caption", None),
                    headers=headers,
                    rows=rows,
                    parse_confidence=0.9,
                )
            )
        return DoclingOutput(markdown=str(document.export_to_markdown()), tables=tables)
