"""Docling client: layout-aware tables, figures, and Markdown body text.

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
from lattice.ingestion.models import ParsedDocument, TableArtifact

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

    def extract_tables(self, pdf_path: str) -> list[TableArtifact]:  # pragma: no cover
        """Extract structured tables from a PDF via Docling TableFormer."""
        if not self.available():
            log.warning("docling.unavailable", path=pdf_path)
            return []
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        tables: list[TableArtifact] = []
        for i, table in enumerate(getattr(result.document, "tables", [])):
            try:
                df = table.export_to_dataframe()
                headers = [str(c) for c in df.columns]
                rows = [[str(v) for v in row] for row in df.to_numpy().tolist()]
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
        return tables

    def to_markdown(self, pdf_path: str) -> str:  # pragma: no cover
        if not self.available():
            return ""
        from docling.document_converter import DocumentConverter

        return str(DocumentConverter().convert(pdf_path).document.export_to_markdown())


def apply_reconciliation(
    doc: ParsedDocument, docling_sections: dict[str, str], settings: DoclingSettings
) -> ParsedDocument:
    """Attach reconciled parse_confidence to each prose section in ``doc``.

    ``docling_sections`` maps section_id -> Docling text for the same region. The
    document's ``overall_confidence`` becomes the minimum across sections.
    """
    min_conf = 1.0
    for section in doc.sections:
        dtext = docling_sections.get(section.section_id)
        if dtext is None:
            continue
        decision = reconcile(section.text, dtext, settings.reconcile_threshold)
        section.parse_confidence = decision.confidence
        if decision.accept and decision.text:
            section.text = decision.text
        min_conf = min(min_conf, decision.confidence)
    doc.overall_confidence = min_conf
    return doc
