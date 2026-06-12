"""GROBID client + TEI parser.

GROBID is the backbone for document structure, metadata, and references. The
network call (``process_fulltext``) is thin; the value is in :func:`parse_tei`,
a pure function that turns GROBID's TEI XML into a :class:`ParsedDocument`. Keeping
it pure makes it testable against fixture XML with no GROBID server.
"""

from __future__ import annotations

import re

import httpx

from lattice.config import GrobidSettings
from lattice.core.errors import ParserError, ParserTimeoutError
from lattice.core.hashing import stable_id
from lattice.ingestion.models import (
    ParsedDocument,
    ParsedReference,
    ParsedSection,
    RegionType,
)

TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"t": TEI_NS}
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _text(el: object) -> str:
    """Collapsed text content of an lxml element (or empty string)."""
    if el is None:
        return ""
    from lxml import etree  # local import: lxml is an optional/dev dependency

    raw = etree.tostring(el, method="text", encoding="unicode")
    return re.sub(r"\s+", " ", raw).strip()


def _first_year(text: str) -> int | None:
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else None


def _persname(el: object) -> str:
    from lxml import etree  # noqa: F401

    forenames = el.findall(".//t:persName/t:forename", _NS) or el.findall(  # type: ignore[attr-defined]
        ".//t:forename", _NS
    )
    surname = el.find(".//t:surname", _NS)  # type: ignore[attr-defined]
    parts = [_text(f) for f in forenames] + ([_text(surname)] if surname is not None else [])
    return " ".join(p for p in parts if p)


def parse_tei(xml: str | bytes) -> ParsedDocument:
    """Parse GROBID TEI XML into a ParsedDocument. Pure and dependency-light."""
    from lxml import etree

    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    try:
        root = etree.fromstring(xml)
    except etree.XMLSyntaxError as exc:  # pragma: no cover - defensive
        raise ParserError(f"invalid TEI XML: {exc}") from exc

    # --- Header: title, authors, abstract, doi, year, venue ---
    title_el = root.find(".//t:teiHeader//t:titleStmt/t:title", _NS)
    title = _text(title_el)

    authors: list[str] = []
    for author in root.findall(".//t:teiHeader//t:sourceDesc//t:author", _NS):
        name = _persname(author)
        if name:
            authors.append(name)
    # Deduplicate preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for a in authors:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    authors = deduped

    abstract_el = root.find(".//t:profileDesc//t:abstract", _NS)
    abstract = _text(abstract_el) or None

    doi_el = root.find('.//t:idno[@type="DOI"]', _NS)
    doi = _text(doi_el) or None

    venue_el = root.find(".//t:sourceDesc//t:monogr/t:title", _NS)
    venue = _text(venue_el) or None

    date_el = root.find(".//t:publicationStmt/t:date", _NS)
    year = _first_year(_text(date_el)) if date_el is not None else None
    if year is None and date_el is not None:
        year = _first_year(date_el.get("when", ""))

    # --- Body sections ---
    sections: list[ParsedSection] = []
    for i, div in enumerate(root.findall(".//t:text/t:body/t:div", _NS)):
        head = div.find("t:head", _NS)
        head_text = _text(head)
        paras = [_text(p) for p in div.findall("t:p", _NS)]
        body = "\n".join(p for p in paras if p)
        if not body and not head_text:
            continue
        section_title = head_text or f"Section {i + 1}"
        sections.append(
            ParsedSection(
                section_id=stable_id("grobid_section", section_title, str(i)),
                title=section_title,
                text=body,
                level=1,
                region_type=RegionType.PROSE,
                parse_confidence=1.0,
            )
        )

    # --- References ---
    references: list[ParsedReference] = []
    for bibl in root.findall(".//t:text/t:back//t:listBibl/t:biblStruct", _NS):
        rtitle = _text(bibl.find(".//t:analytic/t:title", _NS)) or _text(
            bibl.find(".//t:monogr/t:title", _NS)
        )
        rdoi = _text(bibl.find('.//t:idno[@type="DOI"]', _NS)) or None
        rarxiv = _text(bibl.find('.//t:idno[@type="arXiv"]', _NS)) or None
        rdate = bibl.find(".//t:date", _NS)
        ryear = None
        if rdate is not None:
            ryear = _first_year(rdate.get("when", "")) or _first_year(_text(rdate))
        rauthors = [
            _persname(a) for a in bibl.findall(".//t:author", _NS) if _persname(a)
        ]
        if rtitle or rdoi or rarxiv:
            references.append(
                ParsedReference(
                    raw=_text(bibl)[:500],
                    title=rtitle or None,
                    doi=rdoi,
                    arxiv_id=rarxiv,
                    year=ryear,
                    authors=rauthors,
                )
            )

    return ParsedDocument(
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        venue=venue,
        doi=doi,
        sections=sections,
        references=references,
        overall_confidence=1.0 if sections else 0.5,
    )


class GrobidClient:
    """Async wrapper around a GROBID ``processFulltextDocument`` endpoint."""

    def __init__(self, settings: GrobidSettings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client

    async def process_fulltext(self, pdf_bytes: bytes, filename: str = "paper.pdf") -> ParsedDocument:
        client = self._client or httpx.AsyncClient(timeout=self._settings.timeout_s)
        owns = self._client is None
        url = f"{self._settings.url.rstrip('/')}/api/processFulltextDocument"
        files = {"input": (filename, pdf_bytes, "application/pdf")}
        data = {
            "consolidateHeader": str(self._settings.consolidate_header),
            "consolidateCitations": str(self._settings.consolidate_citations),
            "includeRawCitations": "1",
        }
        try:
            resp = await client.post(url, files=files, data=data)
            resp.raise_for_status()
            return parse_tei(resp.content)
        except httpx.TimeoutException as exc:
            raise ParserTimeoutError(f"GROBID timed out after {self._settings.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise ParserError(f"GROBID request failed: {exc}") from exc
        finally:
            if owns:
                await client.aclose()
