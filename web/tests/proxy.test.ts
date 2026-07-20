import { afterEach, describe, expect, it, vi } from "vitest";

import { proxyRequest } from "../lib/server-proxy";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("API proxy", () => {
  it("injects the server token and forwards workspace, query, and body", async () => {
    const upstream = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        Response.json({ status: "queued" }, { status: 202 }),
    );
    vi.stubGlobal("fetch", upstream);
    const request = new Request("http://web.test/api/ingest/file?source=ui", {
      method: "POST",
      headers: {
        authorization: "Bearer browser-controlled",
        cookie: "session=browser-secret",
        "content-type": "application/json",
        "x-forwarded-for": "198.51.100.1",
        "x-workspace-id": "research-2026",
      },
      body: JSON.stringify({ paper: "x" }),
    });

    const response = await proxyRequest(request, ["ingest", "file"], {
      LATTICE_API_BASE: "http://api:8000",
      LATTICE_AUTH_TOKEN: "server-secret",
    });

    expect(response.status).toBe(202);
    expect(upstream).toHaveBeenCalledOnce();
    const [url, init] = upstream.mock.calls[0];
    expect(String(url)).toBe("http://api:8000/ingest/file?source=ui");
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe("Bearer server-secret");
    expect(headers.get("cookie")).toBeNull();
    expect(headers.get("x-forwarded-for")).toBeNull();
    expect(headers.get("x-workspace-id")).toBe("research-2026");
    expect(init?.signal).toBe(request.signal);
    expect(new TextDecoder().decode(init?.body as ArrayBuffer)).toBe(
      '{"paper":"x"}',
    );
  });

  it("streams binary and event responses without changing status or content type", async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(bytes, {
            status: 206,
            headers: {
              "content-type": "application/pdf",
              "content-range": "bytes 0-3/4",
            },
          }),
      ),
    );

    const response = await proxyRequest(
      new Request("http://web.test/api/papers/p1/pdf"),
      ["papers", "p1", "pdf"],
      { LATTICE_API_BASE: "http://api:8000" },
    );

    expect(response.status).toBe(206);
    expect(response.headers.get("content-type")).toBe("application/pdf");
    expect(response.headers.get("content-range")).toBe("bytes 0-3/4");
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
  });

  it("returns a typed 502 response when the backend cannot be reached", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new Error("connection refused"))),
    );

    const response = await proxyRequest(
      new Request("http://web.test/api/health"),
      ["health"],
      { LATTICE_API_BASE: "http://api:8000" },
    );

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      detail: "backend unavailable",
    });
  });

  it("rejects an advertised oversized body before reading or forwarding it", async () => {
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);
    const request = new Request("http://web.test/api/ingest/file", {
      method: "POST",
      headers: { "content-length": "1048577" },
      body: "small",
    });

    const response = await proxyRequest(request, ["ingest", "file"], {
      LATTICE_PROXY_MAX_BODY_MB: "1",
    });

    expect(response.status).toBe(413);
    await expect(response.json()).resolves.toEqual({
      detail: "request body exceeds proxy limit",
    });
    expect(upstream).not.toHaveBeenCalled();
  });

  it("rejects a body that exceeds the cap without a size header", async () => {
    const upstream = vi.fn();
    vi.stubGlobal("fetch", upstream);
    const request = new Request("http://web.test/api/query", {
      method: "POST",
      body: "too large",
    });

    const response = await proxyRequest(request, ["query"], {
      LATTICE_PROXY_MAX_BODY_MB: "0.000001",
    });

    expect(response.status).toBe(413);
    expect(upstream).not.toHaveBeenCalled();
  });
});
