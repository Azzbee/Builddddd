import { afterEach, describe, expect, it, vi } from "vitest";

import { api, ApiError, streamQuery } from "../lib/api";

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

describe("streaming query client", () => {
  it("parses CRLF-delimited server-sent events", async () => {
    const payload =
      'event: status\r\ndata: {"query_class":"factual"}\r\n\r\n' +
      'event: final\r\ndata: {"answer":"Grounded answer","confidence":0.9}\r\n\r\n';
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const bytes = new TextEncoder().encode(payload);
        // Split inside the first CRLF pair to cover real network chunking.
        const split = payload.indexOf("\r\n") + 1;
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body, { status: 200 })),
    );

    const events: Array<{ type: string; data: Record<string, unknown> }> = [];
    const done = new Promise<{ sawFinal: boolean; error: string | null }>(
      (resolve) => {
        streamQuery(
          "question",
          (type, data) => events.push({ type, data }),
          resolve,
        );
      },
    );

    await expect(done).resolves.toEqual({ sawFinal: true, error: null });
    expect(events).toEqual([
      { type: "status", data: { query_class: "factual" } },
      {
        type: "final",
        data: { answer: "Grounded answer", confidence: 0.9 },
      },
    ]);
  });
});

describe("contradictions API client", () => {
  it("fetches persisted relations filtered by relation type", async () => {
    const edge = {
      source_id: "claim-a",
      target_id: "claim-b",
      source_paper: "paper-a",
      target_paper: "paper-b",
      relation: "CONTRADICTS",
      confidence: 0.87,
      source_text: "Transformers improve forecasting accuracy",
      target_text: "Transformers show no significant improvement",
      source_evidence: "Table 3",
      target_evidence: "Table 1",
    };
    const request = vi.fn(async () => Response.json([edge], { status: 200 }));
    vi.stubGlobal("fetch", request);

    // The contradictions page calls this on mount so already-persisted
    // relations render without re-running the analysis pass.
    await expect(api.contradictions("CONTRADICTS")).resolves.toEqual([edge]);
    expect(request).toHaveBeenCalledWith(
      "/api/contradictions?relation=CONTRADICTS",
      { cache: "no-store", headers: { "X-Workspace-Id": "default" } },
    );
  });
});

describe("coverage API client", () => {
  const probe = {
    text: "How does diffusion perform on lme copper?",
    source: "facet_cross" as const,
    salience: 0.4,
    origin_paper_ids: [],
    facet_cell: ["diffusion", "lme copper"] as [string, string],
    coverage: 0.12,
    state: "uncovered" as const,
    components: { retrieval: 0.2, support: 0, grounding: 0 },
    pressure: 0.61,
    supporting_papers: [],
    missing_terms: ["diffusion", "copper"],
    best_evidence: null,
  };

  it("requests the probe bank with explicit caps and no global signal", async () => {
    const report = {
      row_facet: "method",
      col_facet: "dataset",
      summary: {
        probe_count: 1,
        coverage_index: 0.12,
        blind_spot_ratio: 1,
        by_state: { covered: 0, partial: 0, uncovered: 1 },
        by_source: { facet_cross: 1 },
      },
      probes: [probe],
      blind_spots: [probe],
      generated: { facet_cross: 3 },
      dropped_by_cap: { facet_cross: 2 },
    };
    const request = vi.fn(async () => Response.json(report, { status: 200 }));
    vi.stubGlobal("fetch", request);

    await expect(api.coverage()).resolves.toEqual(report);
    expect(request).toHaveBeenCalledWith(
      "/api/landscape/coverage?row_facet=method&col_facet=dataset" +
        "&limit=48&blind_spot_limit=10&use_global=false",
      { cache: "no-store", headers: { "X-Workspace-Id": "default" } },
    );
  });

  it("passes the caller's facets, caps, and global-signal choice through", async () => {
    const request = vi.fn(async () =>
      Response.json(
        {
          row_facet: "method",
          col_facet: "concept",
          summary: {
            probe_count: 0,
            coverage_index: 0,
            blind_spot_ratio: 0,
            by_state: {},
            by_source: {},
          },
          probes: [],
          blind_spots: [],
          generated: {},
          dropped_by_cap: {},
        },
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", request);

    await api.coverage("method", "concept", 12, 3, true);
    expect(request).toHaveBeenCalledWith(
      "/api/landscape/coverage?row_facet=method&col_facet=concept" +
        "&limit=12&blind_spot_limit=3&use_global=true",
      { cache: "no-store", headers: { "X-Workspace-Id": "default" } },
    );
  });
});
