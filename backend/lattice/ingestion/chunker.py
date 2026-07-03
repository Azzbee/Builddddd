"""Section-aware chunking.

Chunks never cross section boundaries, so every chunk carries a precise section
anchor for citation grounding. Within a section, text is packed into overlapping
windows sized by an approximate token budget (≈4 chars/token), splitting on
sentence boundaries where possible to avoid cutting mid-sentence.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from lattice.core.hashing import stable_id
from lattice.ingestion.models import ParsedDocument, RegionType

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9])")
_CHARS_PER_TOKEN = 4


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    paper_id: str
    section_id: str
    section_title: str
    ordinal: int  # position within the document
    text: str
    char_start: int  # offset within the section text
    char_end: int
    region_type: RegionType = RegionType.PROSE
    #: 1-based PDF page the source section starts on, when known (for deep-linking).
    page: int | None = None

    @property
    def approx_tokens(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _pack(sentences: list[str], max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    """Pack sentences into windows. Returns (text, char_start, char_end) triples.

    Offsets are approximate (reconstructed from the joined text) but monotonic and
    good enough to deep-link a reader to the right place in a section.
    """
    windows: list[tuple[str, int, int]] = []
    cur: list[str] = []
    cur_len = 0
    cursor = 0  # running char offset across the section

    def flush(start: int) -> None:
        if not cur:
            return
        text = " ".join(cur).strip()
        windows.append((text, start, start + len(text)))

    start_offset = 0
    for sent in sentences:
        slen = len(sent) + 1
        # A single oversized sentence becomes its own (hard-split) window. Flush any
        # open window FIRST so the oversized sentence is split even when it arrives
        # mid-window (previously it was appended whole to a non-empty window and
        # emitted over-budget, blowing the embedder's token limit).
        if slen > max_chars:
            if cur:
                flush(start_offset)
                cur = []
                cur_len = 0
            for piece_start in range(0, len(sent), max_chars):
                piece = sent[piece_start : piece_start + max_chars]
                windows.append((piece, cursor + piece_start, cursor + piece_start + len(piece)))
            cursor += len(sent) + 1
            start_offset = cursor
            continue
        if cur_len + slen > max_chars and cur:
            flush(start_offset)
            # Build overlap tail for the next window.
            tail: list[str] = []
            tail_len = 0
            for s in reversed(cur):
                if tail_len + len(s) > overlap_chars:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            cur = tail
            cur_len = tail_len
            start_offset = max(0, cursor - tail_len)
        cur.append(sent)
        cur_len += slen
        cursor += slen
    flush(start_offset)
    return windows


def chunk_document(
    doc: ParsedDocument,
    paper_id: str,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_chars: int = 40,
) -> list[Chunk]:
    """Produce section-anchored chunks for a parsed document.

    The abstract is emitted as its own chunk (high-signal, frequently retrieved).
    Tables/formulas/figures are not chunked here; they live as structured
    artifacts and are linked to Results during extraction.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    sections: list[tuple[str, str, str, RegionType, int | None]] = []
    if doc.abstract:
        sections.append(("abstract", "Abstract", doc.abstract, RegionType.PROSE, 1))
    for s in doc.sections:
        if s.region_type in (RegionType.TABLE, RegionType.FORMULA, RegionType.FIGURE):
            continue
        sections.append((s.section_id, s.title, s.text, s.region_type, s.page))

    chunks: list[Chunk] = []
    ordinal = 0
    for section_id, title, text, region, page in sections:
        sentences = _split_sentences(text)
        if not sentences:
            continue
        for win_text, cstart, cend in _pack(sentences, max_chars, overlap_chars):
            if len(win_text) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=stable_id(paper_id, section_id, str(ordinal)),
                    paper_id=paper_id,
                    section_id=section_id,
                    section_title=title,
                    ordinal=ordinal,
                    text=win_text,
                    char_start=cstart,
                    char_end=cend,
                    region_type=region,
                    page=page,
                )
            )
            ordinal += 1
    return chunks
