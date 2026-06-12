from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from lattice.api.deps import Container, get_container, require_auth
from lattice.core.hashing import normalize_arxiv
from lattice.ingestion.models import IngestJob

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_auth)])


class ArxivRequest(BaseModel):
    arxiv_id: str


async def _run_and_store(c: Container, source_ref: str, pdf: bytes) -> IngestJob:
    job = await c.ingestion.ingest_pdf(source_ref, pdf)
    await c.jobs.save(job)
    return job


@router.post("/file")
async def ingest_file(
    file: UploadFile = File(...), c: Container = Depends(get_container)
) -> dict[str, object]:
    # Stage failures are captured as resumable job state, not HTTP errors, so the
    # caller always gets a job back to inspect/retry.
    data = await file.read()
    job = await _run_and_store(c, file.filename or "upload.pdf", data)
    return job.model_dump(mode="json")


@router.post("/arxiv")
async def ingest_arxiv(
    req: ArxivRequest, c: Container = Depends(get_container)
) -> dict[str, object]:
    arxiv_id = normalize_arxiv(req.arxiv_id)
    if not arxiv_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid arxiv id")
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"arxiv fetch failed: {exc}") from exc
    job = await _run_and_store(c, f"arxiv:{arxiv_id}", pdf)
    return job.model_dump(mode="json")


@router.get("/jobs")
async def list_jobs(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    jobs = await c.jobs.all_jobs()
    return [j.model_dump(mode="json") for j in sorted(jobs, key=lambda j: j.created_at, reverse=True)]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, c: Container = Depends(get_container)) -> dict[str, object]:
    job = await c.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job.model_dump(mode="json")
