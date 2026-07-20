import { afterEach, describe, expect, it, vi } from "vitest";

import { api, ApiError } from "../lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ingestion API client", () => {
  it("posts a retry request with an encoded job id", async () => {
    const job = {
      job_id: "job with spaces",
      source_ref: "paper.pdf",
      paper_id: null,
      stage: "parsing",
      status: "queued",
      error_code: null,
      error_message: null,
      retryable: true,
      attempts: 1,
    };
    const request = vi.fn(async () => Response.json(job, { status: 202 }));
    vi.stubGlobal("fetch", request);

    await expect(api.retryJob(job.job_id)).resolves.toEqual(job);
    expect(request).toHaveBeenCalledWith(
      "/api/ingest/jobs/job%20with%20spaces/retry",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Workspace-Id": "default",
        },
        body: "{}",
      },
    );
  });

  it("preserves a backend JSON error for the UI", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json(
          { detail: "job reached the configured attempt cap" },
          { status: 409 },
        ),
      ),
    );

    const error = await api
      .retryJob("job-1")
      .catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      path: "/ingest/jobs/job-1/retry",
      status: 409,
      message: "job reached the configured attempt cap (409)",
    });
  });

  it("uses a bounded text error when a proxy does not return JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(`gateway failure ${"x".repeat(500)}`, { status: 502 }),
      ),
    );

    const error = await api.listPapers().catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(502);
    expect((error as ApiError).message.length).toBeLessThanOrEqual(306);
    expect((error as ApiError).message).toMatch(/^gateway failure/);
  });
});
