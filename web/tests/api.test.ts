import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../lib/api";

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
});
