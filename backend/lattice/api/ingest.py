from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel

from lattice.api.deps import Container, get_container, require_auth
from lattice.core.hashing import normalize_arxiv
from lattice.ingestion.dispatch import JobQueueUnavailable, JobRetryRejected
from lattice.ingestion.models import IngestJob

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_auth)])


class ArxivRequest(BaseModel):
    arxiv_id: str


async def _read_bounded_response(response: httpx.Response, cap: int) -> bytes:
    """Read a remote response without allowing it to exceed ``cap`` bytes."""
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            advertised_size = int(content_length)
        except ValueError:
            advertised_size = None
        if advertised_size is not None and advertised_size > cap:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "remote PDF is too large")

    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > cap:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "remote PDF is too large")
    return bytes(body)


async def _submit(c: Container, source_ref: str, pdf: bytes, response: Response) -> IngestJob:
    assert c.dispatcher is not None
    try:
        job = await c.dispatcher.submit(source_ref, pdf)
    except JobQueueUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    response.status_code = (
        status.HTTP_202_ACCEPTED if c.dispatcher.asynchronous else status.HTTP_200_OK
    )
    return job


@router.post("/file")
async def ingest_file(
    response: Response,
    file: UploadFile = File(...),
    c: Container = Depends(get_container),
) -> dict[str, object]:
    # Reject oversized uploads before buffering the whole body (DoS guard).
    cap = c.settings.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > cap:
        raise HTTPException(
            413,
            f"file exceeds {c.settings.max_upload_mb} MB limit",
        )
    # Stage failures are captured as resumable job state, not HTTP errors, so the
    # caller always gets a job back to inspect/retry.
    data = await file.read()
    if len(data) > cap:
        raise HTTPException(
            413,
            f"file exceeds {c.settings.max_upload_mb} MB limit",
        )
    job = await _submit(c, file.filename or "upload.pdf", data, response)
    return job.model_dump(mode="json")


@router.post("/arxiv")
async def ingest_arxiv(
    req: ArxivRequest,
    response: Response,
    c: Container = Depends(get_container),
) -> dict[str, object]:
    arxiv_id = normalize_arxiv(req.arxiv_id)
    if not arxiv_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid arxiv id")
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    cap = c.settings.max_upload_mb * 1024 * 1024
    try:
        async with (
            httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            pdf = await _read_bounded_response(resp, cap)
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"arxiv fetch failed: {exc}") from exc
    job = await _submit(c, f"arxiv:{arxiv_id}", pdf, response)
    return job.model_dump(mode="json")


@router.get("/jobs")
async def list_jobs(c: Container = Depends(get_container)) -> list[dict[str, object]]:
    jobs = await c.jobs.all_jobs()
    return [
        j.model_dump(mode="json") for j in sorted(jobs, key=lambda j: j.created_at, reverse=True)
    ]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, c: Container = Depends(get_container)) -> dict[str, object]:
    job = await c.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job.model_dump(mode="json")


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    response: Response,
    c: Container = Depends(get_container),
) -> dict[str, object]:
    assert c.dispatcher is not None
    try:
        job = await c.dispatcher.retry(job_id)
    except JobRetryRejected as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except JobQueueUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    response.status_code = (
        status.HTTP_202_ACCEPTED if c.dispatcher.asynchronous else status.HTTP_200_OK
    )
    return job.model_dump(mode="json")
