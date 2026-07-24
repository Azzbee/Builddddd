from __future__ import annotations

from pathlib import Path

import anyio
from lattice.config import DoclingSettings
from lattice.ingestion.docling_client import DoclingOutput
from lattice.ingestion.hybrid_parser import HybridParser, markdown_sections
from lattice.ingestion.models import ParsedDocument, ParsedSection, TableArtifact


class Primary:
    async def process_fulltext(
        self, pdf_bytes: bytes, filename: str = "paper.pdf"
    ) -> ParsedDocument:
        return ParsedDocument(
            title="Paper",
            sections=[
                ParsedSection(
                    section_id="s1",
                    title="Results",
                    text="GROBID result says RMSE is 0.21.",
                    page=2,
                )
            ],
        )


class Layout:
    def __init__(self, output: DoclingOutput, *, available: bool = True) -> None:
        self.output = output
        self.is_available = available
        self.path: str | None = None
        self.existed_during_extract = False

    def available(self) -> bool:
        return self.is_available

    def extract(self, pdf_path: str) -> DoclingOutput:
        self.path = pdf_path
        self.existed_during_extract = Path(pdf_path).exists()
        return self.output


class Vision:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[bytes, str]] = []

    async def describe_image(self, image_png: bytes, prompt: str) -> str:
        self.calls.append((image_png, prompt))
        return self.answer


def test_markdown_sections_normalizes_headings() -> None:
    assert markdown_sections("# Results\nRMSE 0.12\n\n## Future Work\nMore data") == {
        "results": "RMSE 0.12",
        "future work": "More data",
    }


async def test_hybrid_parser_uses_docling_text_tables_and_cleans_temp_file() -> None:
    table = TableArtifact(table_id="t1", headers=["Model"], rows=[["LSTM"]])
    layout = Layout(
        DoclingOutput(
            markdown="# Results\nGROBID result says RMSE is 0.21 with more detail.",
            tables=[table],
        )
    )
    parser = HybridParser(Primary(), DoclingSettings(), layout=layout, vision=None)

    document = await parser.process_fulltext(b"%PDF", "paper.pdf")

    assert document.sections[0].text.endswith("with more detail.")
    assert document.tables == [table]
    assert layout.existed_during_extract
    assert layout.path is not None and not await anyio.Path(layout.path).exists()


async def test_hybrid_parser_arbitrates_low_agreement_with_page_image() -> None:
    layout = Layout(DoclingOutput(markdown="# Results\nDocling reports MAE 0.09.", tables=[]))
    vision = Vision("Ground truth RMSE is 0.12.")
    rendered: list[int] = []

    def render(_pdf: bytes, page: int) -> bytes:
        rendered.append(page)
        return b"png"

    parser = HybridParser(
        Primary(),
        DoclingSettings(reconcile_threshold=0.95),
        layout=layout,
        vision=vision,
        page_renderer=render,
    )
    document = await parser.process_fulltext(b"%PDF", "paper.pdf")

    assert document.sections[0].text == "Ground truth RMSE is 0.12."
    assert document.sections[0].parse_confidence == 1.0
    assert document.overall_confidence == 1.0
    assert rendered == [2]
    assert vision.calls and vision.calls[0][0] == b"png"
    assert "GROBID result says RMSE is 0.21" in vision.calls[0][1]
    assert "Docling reports MAE 0.09" in vision.calls[0][1]


async def test_hybrid_parser_keeps_low_confidence_text_without_vision() -> None:
    layout = Layout(
        DoclingOutput(
            markdown="# Results\nA much longer unrelated Docling passage with extra evidence.",
            tables=[],
        )
    )
    parser = HybridParser(
        Primary(),
        DoclingSettings(vision_enabled=False, reconcile_threshold=0.95),
        layout=layout,
    )
    document = await parser.process_fulltext(b"%PDF")

    assert document.sections[0].text.startswith("A much longer")
    assert document.sections[0].parse_confidence < 0.95
    assert document.overall_confidence < 0.95


async def test_hybrid_parser_skips_unavailable_layout_parser() -> None:
    layout = Layout(DoclingOutput(markdown="# Results\nreplacement", tables=[]), available=False)
    document = await HybridParser(Primary(), DoclingSettings(), layout=layout).process_fulltext(
        b"%PDF"
    )
    assert document.sections[0].text == "GROBID result says RMSE is 0.21."
    assert layout.path is None
