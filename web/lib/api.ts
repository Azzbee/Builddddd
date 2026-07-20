import type {
  AgentAnswer,
  GraphData,
  GraphDelta,
  GraphTimeline,
  IngestJob,
  MatrixCell,
  PaperCard,
  PaperSummary,
  PdfMeta,
  ResearchProposal,
  SelectionSummary,
} from "./types";
import { getWorkspace } from "./workspace";

// All requests go through the Next.js /api proxy (see next.config.mjs).
const BASE = "/api";
const MAX_ERROR_DETAIL_LENGTH = 300;

export class ApiError extends Error {
  constructor(
    readonly path: string,
    readonly status: number,
    detail: string,
  ) {
    super(`${detail} (${status})`);
    this.name = "ApiError";
  }
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  return { "X-Workspace-Id": getWorkspace(), ...extra };
}

function boundedDetail(value: string): string {
  return value.trim().slice(0, MAX_ERROR_DETAIL_LENGTH);
}

async function apiError(response: Response, path: string): Promise<ApiError> {
  let detail = "request failed";
  try {
    const body: unknown = await response.clone().json();
    if (
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      typeof body.detail === "string"
    ) {
      detail = boundedDetail(body.detail) || detail;
    }
  } catch {
    try {
      detail = boundedDetail(await response.text()) || detail;
    } catch {
      // Keep the stable fallback when the response body cannot be decoded.
    }
  }
  return new ApiError(path, response.status, detail);
}

async function json<T>(response: Response, path: string): Promise<T> {
  if (!response.ok) throw await apiError(response, path);
  return response.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    cache: "no-store",
    headers: headers(),
  });
  return json<T>(res, path);
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  return json<T>(res, path);
}

export const api = {
  listPapers: () => get<PaperSummary[]>("/papers"),
  getPaper: (id: string) => get<PaperCard>(`/papers/${encodeURIComponent(id)}`),
  summarizePapers: (paper_ids: string[]) =>
    post<SelectionSummary>("/papers/summarize", { paper_ids }),
  pdfMeta: (id: string) =>
    get<PdfMeta>(`/papers/${encodeURIComponent(id)}/pdf/meta`),
  // Fetch the PDF through the workspace-aware proxy and return an object URL the
  // embedded viewer can use (an <iframe src> can't carry the X-Workspace-Id header).
  async pdfObjectUrl(id: string): Promise<string> {
    const res = await fetch(`${BASE}/papers/${encodeURIComponent(id)}/pdf`, {
      cache: "no-store",
      headers: headers(),
    });
    if (!res.ok) throw await apiError(res, `/papers/${id}/pdf`);
    return URL.createObjectURL(await res.blob());
  },
  graph: (minWeight = 0) => get<GraphData>(`/graph?min_weight=${minWeight}`),
  graphAsOf: (asOfYear: number, minWeight = 0) =>
    get<GraphData>(`/graph?min_weight=${minWeight}&as_of_year=${asOfYear}`),
  graphTimeline: () => get<GraphTimeline>("/graph/timeline"),
  graphDelta: (sinceYear: number, untilYear?: number) =>
    get<GraphDelta>(
      `/graph/delta?since_year=${sinceYear}${untilYear != null ? `&until_year=${untilYear}` : ""}`,
    ),
  graphStats: () =>
    get<{ papers: number; edges: number; communities: number }>("/graph/stats"),
  query: (question: string) => post<AgentAnswer>("/query", { question }),
  matrix: (row: string, col: string) =>
    get<{
      row_facet: string;
      col_facet: string;
      cells: MatrixCell[];
      top_gaps: MatrixCell[];
    }>(`/landscape/matrix?row=${row}&col=${col}`),
  momentum: () =>
    get<{ movers: Record<string, unknown>[] }>("/landscape/momentum"),
  opportunities: (
    rowFacet = "method",
    colFacet = "dataset",
    limit = 6,
    useGlobal = true,
  ) =>
    get<{
      row_facet: string;
      col_facet: string;
      proposals: ResearchProposal[];
    }>(
      `/landscape/opportunities?row_facet=${rowFacet}&col_facet=${colFacet}&limit=${limit}&use_global=${useGlobal}`,
    ),
  proposal: (
    row: string,
    col: string,
    rowFacet = "method",
    colFacet = "dataset",
    useGlobal = true,
  ) =>
    get<ResearchProposal>(
      `/landscape/proposal?row=${encodeURIComponent(row)}&col=${encodeURIComponent(col)}` +
        `&row_facet=${rowFacet}&col_facet=${colFacet}&use_global=${useGlobal}`,
    ),
  quadrants: () =>
    get<{
      known_knowns: Record<string, unknown>[];
      known_unknowns: Record<string, unknown>[];
      unknown_knowns: Record<string, unknown>[];
    }>("/landscape/quadrants"),
  analyzeContradictions: () =>
    post<{ analyzed: number; contradictions: number; supports: number }>(
      "/contradictions/analyze",
      {},
    ),
  contradictions: (relation = "CONTRADICTS") =>
    get<Record<string, unknown>[]>(`/contradictions?relation=${relation}`),
  lineage: (method: string) =>
    get<{
      method: string;
      nodes: { id: string; title: string; year: number | null }[];
      edges: { source: string; target: string; kind: string }[];
      timeline: Record<string, string[]>;
    }>(`/lineage?method=${encodeURIComponent(method)}`),
  readingQueue: () =>
    get<{ read_count: number; queue: Record<string, unknown>[] }>(
      "/reading-queue",
    ),
  relatedWork: () =>
    get<{
      clusters: Record<string, unknown>[];
      markdown: string;
      bibtex: string;
    }>("/related-work"),
  generateDigest: () => post<{ markdown: string }>("/digest/generate", {}),
  watchQueue: () => get<Record<string, unknown>[]>("/watch/queue"),
  approveWatch: (arxiv_id: string, approve: boolean) =>
    post<{ arxiv_id: string; status: string }>("/watch/approve", {
      arxiv_id,
      approve,
    }),
  jobs: () => get<IngestJob[]>("/ingest/jobs"),
  retryJob: (jobId: string) =>
    post<IngestJob>(`/ingest/jobs/${encodeURIComponent(jobId)}/retry`, {}),
  ingestArxiv: (arxiv_id: string) =>
    post<IngestJob>("/ingest/arxiv", { arxiv_id }),
  async ingestFile(file: File): Promise<IngestJob> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/ingest/file`, {
      method: "POST",
      body: form,
      headers: headers(),
    });
    return json<IngestJob>(res, "/ingest/file");
  },
};

// SSE streaming for the chat agent.
//
// The backend guarantees the stream always ends with exactly one `final` event.
// `onDone` still fires on every terminal path — normal completion, an aborted
// request, or a network drop the server never saw — so the caller can always
// reset its UI (e.g. clear a "thinking..." spinner) instead of hanging forever.
// `saw_final` tells the caller whether a real answer arrived; if not, `error`
// carries the reason (or null on a clean cancel).
export function streamQuery(
  question: string,
  onEvent: (type: string, data: Record<string, unknown>) => void,
  onDone?: (info: { sawFinal: boolean; error: string | null }) => void,
): () => void {
  const controller = new AbortController();
  let sawFinal = false;
  (async () => {
    const res = await fetch(`${BASE}/query/stream`, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!res.ok) throw await apiError(res, "/query/stream");
    if (!res.body) throw new Error("query/stream -> empty body");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    // Reset the event type per frame: a frame with only a `data:` line defaults
    // to "message" rather than inheriting the previous frame's event name.
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        let eventType = "message";
        let data: string | null = null;
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) eventType = line.slice(6).trim();
          else if (line.startsWith("data:")) data = line.slice(5).trim();
        }
        if (data === null) continue;
        try {
          const parsed = JSON.parse(data);
          // Mark sawFinal only after a successful parse+dispatch: a corrupted final
          // frame must NOT suppress the caller's recoverable-error fallback.
          onEvent(eventType, parsed);
          if (eventType === "final") sawFinal = true;
        } catch {
          /* ignore malformed frames */
        }
      }
    }
  })()
    .then(() => onDone?.({ sawFinal, error: null }))
    .catch((err: unknown) => {
      // An aborted request is a deliberate cancel, not an error.
      const aborted = err instanceof DOMException && err.name === "AbortError";
      onDone?.({ sawFinal, error: aborted ? null : String(err) });
    });
  return () => controller.abort();
}
