"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { GraphData, GraphNode, SelectionSummary } from "@/lib/types";

// Sigma touches the DOM/WebGL, so load it client-only.
const GraphExplorer = dynamic(
  () => import("@/components/GraphExplorer").then((m) => m.GraphExplorer),
  { ssr: false },
);

export default function GraphPage() {
  const [data, setData] = useState<GraphData>({ nodes: [], edges: [] });
  const [minWeight, setMinWeight] = useState(0.35);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [search, setSearch] = useState("");
  const [error, setError] = useState<string>();
  const [lasso, setLasso] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [summary, setSummary] = useState<SelectionSummary | null>(null);
  const [summarizing, setSummarizing] = useState(false);

  useEffect(() => {
    api.graph(0).then(setData).catch((e) => setError(String(e)));
  }, []);

  const filtered = useMemo<GraphData>(
    // Keep all nodes visible (even isolated ones) so a sparse corpus still shows.
    () => ({ nodes: data.nodes, edges: data.edges.filter((e) => e.weight >= minWeight) }),
    [data, minWeight],
  );

  async function summarizeSelection(ids: string[]) {
    setSelectedIds(new Set(ids));
    setSummary(null);
    setSummarizing(true);
    try {
      setSummary(await api.summarizePapers(ids));
    } catch (e) {
      setError(String(e));
    } finally {
      setSummarizing(false);
    }
  }

  function clearSelection() {
    setSelectedIds(new Set());
    setSummary(null);
  }

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">Graph explorer</h1>
          <p className="text-sm text-muted">
            Size = centrality, color = community, edge thickness = weight.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <input
            className="input max-w-[200px]"
            placeholder="Search papers..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button
            className={lasso ? "btn-accent" : "btn"}
            onClick={() => {
              setLasso((v) => !v);
              clearSelection();
            }}
            title="Drag a box over the graph to select and summarize a cluster"
          >
            {lasso ? "Lasso: on" : "Lasso select"}
          </button>
          <label className="flex items-center gap-2 text-sm text-muted">
            min weight {minWeight.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={minWeight}
              onChange={(e) => setMinWeight(parseFloat(e.target.value))}
            />
          </label>
        </div>
      </header>

      {error && <div className="card border-bad text-bad">Backend unreachable: {error}</div>}
      {lasso && (
        <div className="text-xs text-muted">
          Drag a rectangle over the graph to select papers, then read the cluster brief on the right.
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_360px]">
        <GraphExplorer
          data={filtered}
          onSelect={setSelected}
          highlight={search}
          lassoMode={lasso}
          selectedIds={selectedIds}
          onLassoSelect={summarizeSelection}
        />
        <aside className="card h-fit">
          {selectedIds.size > 0 ? (
            <SelectionPanel
              summary={summary}
              loading={summarizing}
              count={selectedIds.size}
              onClear={clearSelection}
            />
          ) : selected ? (
            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-white">{selected.title}</h2>
              <div className="flex flex-wrap gap-1">
                <span className="chip">community {selected.community}</span>
                <span className="chip">centrality {selected.centrality.toFixed(2)}</span>
                {selected.year && <span className="chip">{selected.year}</span>}
                {selected.needs_review && (
                  <span className="chip border-warn text-warn">needs review</span>
                )}
              </div>
              <Link href={`/papers/${selected.id}`} className="btn-accent inline-block">
                Open paper card
              </Link>
            </div>
          ) : (
            <p className="text-sm text-muted">
              Click a node to inspect it, or use <span className="text-ink">Lasso select</span> to
              summarize a cluster.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}

function SelectionPanel({
  summary,
  loading,
  count,
  onClear,
}: {
  summary: SelectionSummary | null;
  loading: boolean;
  count: number;
  onClear: () => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Selection brief</h2>
        <button className="text-xs text-muted hover:text-ink" onClick={onClear}>
          clear
        </button>
      </div>
      {loading && <p className="text-sm text-muted">Summarizing {count} papers...</p>}
      {!loading && summary && (
        <div className="space-y-3 text-sm">
          <div className="flex flex-wrap gap-1">
            <span className="chip">{summary.count} papers</span>
            {summary.year_range && (
              <span className="chip">
                {summary.year_range[0]}–{summary.year_range[1]}
              </span>
            )}
            {summary.domains.map((d) => (
              <span key={d} className="chip">
                {d}
              </span>
            ))}
          </div>

          {summary.methods.length > 0 && (
            <Group label="Shared methods" items={summary.methods} />
          )}
          {summary.datasets.length > 0 && (
            <Group label="Shared datasets" items={summary.datasets} />
          )}

          {summary.contradictions.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-semibold text-warn">Tensions in selection</div>
              <ul className="space-y-1 text-xs text-ink">
                {summary.contradictions.map((c, i) => (
                  <li key={i} className="rounded border border-border bg-panel2 p-2">
                    <span className="text-muted">{c.source_text}</span> vs.{" "}
                    <span className="text-muted">{c.target_text}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {summary.open_problems.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-semibold text-white">Recurring open problems</div>
              <ul className="list-disc space-y-0.5 pl-4 text-xs text-ink">
                {summary.open_problems.map((p, i) => (
                  <li key={i}>
                    {p.problem} <span className="text-muted">({p.mentions})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <div className="mb-1 text-xs font-semibold text-white">Papers</div>
            <ul className="space-y-0.5 text-xs">
              {summary.papers.map((p) => (
                <li key={p.paper_id}>
                  <Link href={`/papers/${p.paper_id}`} className="text-accent hover:underline">
                    {p.title}
                  </Link>{" "}
                  <span className="text-muted">{p.year ?? ""}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {!loading && summary && summary.count === 0 && (
        <p className="text-sm text-muted">No papers in the selected region.</p>
      )}
    </div>
  );
}

function Group({ label, items }: { label: string; items: string[] }) {
  return (
    <div>
      <div className="mb-1 text-xs font-semibold text-white">{label}</div>
      <div className="flex flex-wrap gap-1">
        {items.map((it) => (
          <span key={it} className="chip">
            {it}
          </span>
        ))}
      </div>
    </div>
  );
}
