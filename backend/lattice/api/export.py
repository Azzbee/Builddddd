from __future__ import annotations

import io
import re
import zipfile

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from lattice.api.deps import Container, get_container, require_auth

router = APIRouter(tags=["export"], dependencies=[Depends(require_auth)])

_SAFE = re.compile(r"[^a-zA-Z0-9 _-]")


@router.get("/related-work")
async def related_work(c: Container = Depends(get_container)) -> dict[str, object]:
    """A grounded related-work draft grouped by community, with BibTeX."""
    return await c.ingestion.related_work()


@router.get("/export/bibtex")
async def export_bibtex(c: Container = Depends(get_container)) -> Response:
    bib = await c.ingestion.export_bibtex()
    return Response(
        content=bib,
        media_type="application/x-bibtex",
        headers={"Content-Disposition": "attachment; filename=lattice.bib"},
    )


@router.get("/export/obsidian")
async def export_obsidian(c: Container = Depends(get_container)) -> Response:
    """A zip of one Markdown note per paper, wiki-linked to mirror the graph."""
    notes = await c.ingestion.export_obsidian()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for title, content in notes.items():
            safe = _SAFE.sub("", title)[:120].strip() or "untitled"
            zf.writestr(f"{safe}.md", content)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=lattice-obsidian.zip"},
    )
