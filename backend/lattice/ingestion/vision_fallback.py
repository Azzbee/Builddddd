"""Vision-LLM fallback for pages where GROBID and Docling disagree or fail.

When a region's reconciliation confidence is low, the page image is sent to a
vision-capable LLM to arbitrate. This module defines the protocol and prompt; the
actual model call goes through the LLM client (LiteLLM) so it stays
provider-agnostic and testable with a fake.
"""

from __future__ import annotations

from typing import Protocol

from lattice.core.logging import get_logger

log = get_logger("ingestion.vision")

ARBITRATION_PROMPT = """You are arbitrating a parse disagreement on one page of a \
scientific paper. Two parsers produced different text for the same region. Using \
the page image as ground truth, return the correct, clean text for this region. \
Preserve numbers, symbols, and table structure exactly. Do not summarize or add \
commentary. If the region is a table, return it as GitHub-flavored Markdown."""


class VisionModel(Protocol):
    async def describe_image(self, image_png: bytes, prompt: str) -> str: ...


async def arbitrate_region(
    model: VisionModel,
    page_image_png: bytes,
    grobid_text: str,
    docling_text: str,
) -> str:
    """Ask the vision model to produce the authoritative text for a region."""
    prompt = (
        f"{ARBITRATION_PROMPT}\n\n"
        f"Parser A text:\n{grobid_text[:4000]}\n\n"
        f"Parser B text:\n{docling_text[:4000]}"
    )
    result = await model.describe_image(page_image_png, prompt)
    log.info("vision.arbitrated", in_len=len(grobid_text) + len(docling_text), out_len=len(result))
    return result.strip()


CAPTION_PROMPT = """Describe this figure from a scientific paper in 1-3 sentences \
suitable as searchable alt text. State what is plotted, the axes or variables, and \
the main takeaway. Be concrete and concise."""


async def caption_figure(model: VisionModel, figure_png: bytes) -> str:
    """Generate grounded alt text for a figure (one cheap call, batched upstream)."""
    return (await model.describe_image(figure_png, CAPTION_PROMPT)).strip()
