"""Production parser combining GROBID, Docling, and vision arbitration."""

from __future__ import annotations

import io
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import anyio

from lattice.config import DoclingSettings
from lattice.core.hashing import normalize_text
from lattice.core.logging import get_logger
from lattice.ingestion.docling_client import DoclingClient, DoclingOutput, reconcile
from lattice.ingestion.models import ParsedDocument
from lattice.ingestion.vision_fallback import VisionModel, arbitrate_region

log = get_logger("ingestion.hybrid_parser")

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")


class PrimaryParser(Protocol):
    async def process_fulltext(self, pdf_bytes: bytes, filename: str = ...) -> ParsedDocument: ...


class LayoutParser(Protocol):
    def available(self) -> bool: ...
    def extract(self, pdf_path: str) -> DoclingOutput: ...


def markdown_sections(markdown: str) -> dict[str, str]:
    """Split Docling Markdown into normalized heading-to-body sections."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in markdown.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = normalize_text(heading.group(1))
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return {
        heading: "\n".join(lines).strip()
        for heading, lines in sections.items()
        if "\n".join(lines).strip()
    }


def _matching_section(title: str, sections: dict[str, str]) -> str | None:
    key = normalize_text(title)
    if key in sections:
        return sections[key]
    matches = [text for heading, text in sections.items() if key in heading or heading in key]
    return max(matches, key=len) if matches else None


def render_pdf_page(pdf_bytes: bytes, page: int) -> bytes:
    """Render one 1-based PDF page to PNG for vision arbitration."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        index = max(0, min(page - 1, len(document) - 1))
        bitmap = document[index].render(scale=1.5)
        image = bitmap.to_pil()
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    finally:
        document.close()


class HybridParser:
    def __init__(
        self,
        primary: PrimaryParser,
        settings: DoclingSettings,
        *,
        layout: LayoutParser | None = None,
        vision: VisionModel | None = None,
        page_renderer: Callable[[bytes, int], bytes] = render_pdf_page,
    ) -> None:
        self._primary = primary
        self._settings = settings
        self._layout = layout or DoclingClient(settings)
        self._vision = vision
        self._page_renderer = page_renderer

    async def process_fulltext(
        self, pdf_bytes: bytes, filename: str = "paper.pdf"
    ) -> ParsedDocument:
        document = await self._primary.process_fulltext(pdf_bytes, filename)
        if not self._settings.enabled or not self._layout.available():
            return document

        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(pdf_bytes)
                path = Path(handle.name)
            output = await anyio.to_thread.run_sync(self._layout.extract, str(path))
        except Exception as exc:
            log.warning("docling.failed", filename=filename, error=str(exc))
            return document
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

        document.tables.extend(output.tables)
        sections = markdown_sections(output.markdown)
        min_confidence = document.overall_confidence
        for section in document.sections:
            docling_text = _matching_section(section.title, sections)
            if not docling_text:
                continue
            grobid_text = section.text
            decision = reconcile(grobid_text, docling_text, self._settings.reconcile_threshold)
            section.parse_confidence = decision.confidence
            min_confidence = min(min_confidence, decision.confidence)
            if decision.accept:
                section.text = decision.text
                continue
            section.text = decision.text
            if self._vision is None or not self._settings.vision_enabled:
                continue
            try:
                page_image = await anyio.to_thread.run_sync(
                    self._page_renderer, pdf_bytes, section.page or 1
                )
                corrected = await arbitrate_region(
                    self._vision, page_image, grobid_text, docling_text
                )
            except Exception as exc:
                log.warning(
                    "vision.failed",
                    filename=filename,
                    section=section.title,
                    error=str(exc),
                )
                continue
            if corrected:
                section.text = corrected
                section.parse_confidence = 1.0

        document.overall_confidence = min(
            [section.parse_confidence for section in document.sections] or [min_confidence]
        )
        return document
