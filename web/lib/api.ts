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

function headers(extra: Record<string, string> = {}): Record<string, string> {
  return { "X-Workspace-Id": getWorkspace(), ...extra };
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store", headers: headers() });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  listPapers: () => get<PaperSummary[]>("/papers"),
  getPaper: (id: string) => get<PaperCard>(`/papers/${encodeURIComponent(id)}`),
  summarizePapers: (paper_ids: string[]) =>
    post<SelectionSummary>("/papers/summarize", { paper_ids }),
  pdfMeta: (id: string) => get<PdfMeta>(`/papers/${encodeURIComponent(id)}/pdf/meta`),
  // Fetch the PDF through the workspace-aware proxy and return an object URL the
  // embedded viewer can use (an <iframe src> can't carry the X-Workspace-Id header).
  async pdfObjectUrl(id: string): Promise<string> {
    const res = await fetch(`${BASE}/papers/${encodeURIComponent(id)}/pdf`, {
      cache: "no-store",
      headers: headers(),
    });
    if (!res.ok) throw new Error(`pdf -> ${res.status}`);
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
  graphStats: () => get<{ papers: number; edges: number; communities: number }>("/graph/stats"),
  query: (question: string) => post<AgentAnswer>("/query", { question }),
  matrix: (row: string, col: string) =>
    get<{ row_facet: string; col_facet: string; cells: MatrixCell[]; top_gaps: MatrixCell[] }>(
      `/landscape/matrix?row=${row}&col=${col}`,
    ),
  momentum: () => get<{ movers: Record<string, unknown>[] }>("/landscape/momentum"),
  opportunities: (rowFacet = "method", colFacet = "dataset", limit = 6, useGlobal = true) =>
    get<{ row_facet: string; col_facet: string; proposals: ResearchProposal[] }>(
      `/landscape/opportunities?row_facet=${rowFacet}&col_facet=${colFacet}&limit=${limit}&use_global=${useGlobal}`,
    ),
  proposal: (row: string, col: string, rowFacet = "method", colFacet = "dataset", useGlobal = true) =>
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
  readingQueue: () => get<{ read_count: number; queue: Record<string, unknown>[] }>("/reading-queue"),
  relatedWork: () =>
    get<{ clusters: Record<string, unknown>[]; markdown: string; bibtex: string }>("/related-work"),
  generateDigest: () => post<{ markdown: string }>("/digest/generate", {}),
  watchQueue: () => get<Record<string, unknown>[]>("/watch/queue"),
  approveWatch: (arxiv_id: string, approve: boolean) =>
    post<{ arxiv_id: string; status: string }>("/watch/approve", { arxiv_id, approve }),
  jobs: () => get<IngestJob[]>("/ingest/jobs"),
  ingestArxiv: (arxiv_id: string) => post<IngestJob>("/ingest/arxiv", { arxiv_id }),
  async ingestFile(file: File): Promise<IngestJob> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/ingest/file`, { method: "POST", body: form, headers: headers() });
    if (!res.ok) throw new Error(`ingest -> ${res.status}`);
    return res.json() as Promise<IngestJob>;
  },
};

// SSE streaming for the chat agent.
export function streamQuery(
  question: string,
  onEvent: (type: string, data: Record<string, unknown>) => void,
): () => void {
  const controller = new AbortController();
  (async () => {
    const res = await fetch(`${BASE}/query/stream`, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!res.body) return;
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventType = "message";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) eventType = line.slice(6).trim();
          else if (line.startsWith("data:")) {
            try {
              onEvent(eventType, JSON.parse(line.slice(5).trim()));
            } catch {
              /* ignore malformed frames */
            }
          }
        }
      }
    }
  })().catch(() => {
    /* aborted or network error */
  });
  return () => controller.abort();
}
