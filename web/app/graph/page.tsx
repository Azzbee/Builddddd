"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { GraphData, GraphNode } from "@/lib/types";

// Sigma touches the DOM/WebGL, so load it client-only.
const GraphExplorer = dynamic(
  () => import("@/components/GraphExplorer").then((m) => m.GraphExplorer),
  { ssr: false },
);

export default function GraphPage() {
  const [data, setData] = useState<GraphData>({ nodes: [], edges: [] });
  const [minWeight, setMinWeight] = useState(0.35);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [error, setError] = useState<string>();

  useEffect(() => {
    api.graph(0).then(setData).catch((e) => setError(String(e)));
  }, []);

  const filtered = useMemo<GraphData>(
    // Keep all nodes visible (even isolated ones) so a sparse corpus still shows.
    () => ({ nodes: data.nodes, edges: data.edges.filter((e) => e.weight >= minWeight) }),
    [data, minWeight],
  );

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">Graph explorer</h1>
          <p className="text-sm text-muted">
            Size = centrality, color = community, edge thickness = weight.
          </p>
        </div>
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
      </header>

      {error && <div className="card border-bad text-bad">Backend unreachable: {error}</div>}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <GraphExplorer data={filtered} onSelect={setSelected} />
        <aside className="card h-fit">
          {selected ? (
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
            <p className="text-sm text-muted">Click a node to inspect it.</p>
          )}
        </aside>
      </div>
    </div>
  );
}
