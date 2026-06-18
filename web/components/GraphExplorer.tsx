"use client";

import { useEffect, useRef, useState } from "react";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import type { GraphData, GraphEdge, GraphNode } from "@/lib/types";

const PALETTE = [
  "#5b8cff", "#9b6bff", "#3fb950", "#d29922", "#f85149",
  "#2dd4bf", "#f472b6", "#a3e635", "#fb923c", "#60a5fa",
];

function colorFor(community: number): string {
  return PALETTE[community % PALETTE.length];
}

export function GraphExplorer({
  data,
  onSelect,
  highlight = "",
}: {
  data: GraphData;
  onSelect: (node: GraphNode | null) => void;
  highlight?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const [hoverEdge, setHoverEdge] = useState<GraphEdge | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.nodes.length === 0) return;
    const graph = new Graph({ multi: false, type: "undirected" });

    data.nodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / data.nodes.length;
      graph.addNode(n.id, {
        label: n.title.length > 48 ? n.title.slice(0, 45) + "..." : n.title,
        x: Math.cos(angle),
        y: Math.sin(angle),
        size: 4 + n.centrality * 14,
        color: colorFor(n.community),
        community: n.community,
        node: n,
      });
    });
    data.edges.forEach((e) => {
      if (graph.hasNode(e.source) && graph.hasNode(e.target) && !graph.hasEdge(e.source, e.target)) {
        graph.addEdge(e.source, e.target, {
          size: 0.5 + e.weight * 4,
          color: "#2a3142",
          weight: e.weight,
          edge: e,
        });
      }
    });

    forceAtlas2.assign(graph, { iterations: 200, settings: { gravity: 1, scalingRatio: 12 } });

    const renderer = new Sigma(graph, containerRef.current, {
      labelColor: { color: "#8b93a7" },
      labelSize: 11,
      defaultEdgeColor: "#2a3142",
      renderEdgeLabels: false,
    });

    renderer.on("clickNode", ({ node }) => onSelect(graph.getNodeAttribute(node, "node")));
    renderer.on("clickStage", () => onSelect(null));
    renderer.on("enterEdge", ({ edge }) => setHoverEdge(graph.getEdgeAttribute(edge, "edge")));
    renderer.on("leaveEdge", () => setHoverEdge(null));

    rendererRef.current = renderer;
    return () => {
      renderer.kill();
      rendererRef.current = null;
    };
  }, [data, onSelect]);

  // Highlight nodes matching the search term; dim the rest.
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const term = highlight.trim().toLowerCase();
    renderer.setSetting("nodeReducer", (_node, attrs) => {
      if (!term) return attrs;
      const match = String(attrs.label || "").toLowerCase().includes(term);
      return match
        ? { ...attrs, zIndex: 1, highlighted: true }
        : { ...attrs, color: "#2a3142", label: "", zIndex: 0 };
    });
    renderer.refresh();
  }, [highlight]);

  return (
    <div className="relative h-[70vh] w-full overflow-hidden rounded-lg border border-border bg-panel">
      <div ref={containerRef} className="h-full w-full" />
      {data.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">
          No graph yet. Ingest papers to populate it.
        </div>
      )}
      {hoverEdge && (
        <div className="absolute bottom-3 left-3 card max-w-xs text-xs">
          <div className="mb-1 font-semibold text-white">Edge anatomy</div>
          <div className="mb-1 text-muted">weight {hoverEdge.weight.toFixed(3)}</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(hoverEdge.components).map(([k, v]) => (
              <span key={k} className="chip">
                {k} {Number(v).toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
