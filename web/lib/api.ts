import type {
  AgentAnswer,
  GraphData,
  IngestJob,
  MatrixCell,
  PaperCard,
  PaperSummary,
} from "./types";

// All requests go through the Next.js /api proxy (see next.config.mjs).
const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  listPapers: () => get<PaperSummary[]>("/papers"),
  getPaper: (id: string) => get<PaperCard>(`/papers/${encodeURIComponent(id)}`),
  graph: (minWeight = 0) => get<GraphData>(`/graph?min_weight=${minWeight}`),
  graphStats: () => get<{ papers: number; edges: number; communities: number }>("/graph/stats"),
  query: (question: string) => post<AgentAnswer>("/query", { question }),
  matrix: (row: string, col: string) =>
    get<{ row_facet: string; col_facet: string; cells: MatrixCell[]; top_gaps: MatrixCell[] }>(
      `/landscape/matrix?row=${row}&col=${col}`,
    ),
  momentum: () => get<{ movers: Record<string, unknown>[] }>("/landscape/momentum"),
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
  jobs: () => get<IngestJob[]>("/ingest/jobs"),
  ingestArxiv: (arxiv_id: string) => post<IngestJob>("/ingest/arxiv", { arxiv_id }),
  async ingestFile(file: File): Promise<IngestJob> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/ingest/file`, { method: "POST", body: form });
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
      headers: { "Content-Type": "application/json" },
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
