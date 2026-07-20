"use client";

import { useEffect, useRef, useState } from "react";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import type { GraphData, GraphEdge, GraphNode } from "@/lib/types";

const PALETTE = [
  "#5b8cff",
  "#9b6bff",
  "#3fb950",
  "#d29922",
  "#f85149",
  "#2dd4bf",
  "#f472b6",
  "#a3e635",
  "#fb923c",
  "#60a5fa",
];

function colorFor(community: number): string {
  return PALETTE[community % PALETTE.length];
}

interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function GraphExplorer({
  data,
  onSelect,
  highlight = "",
  lassoMode = false,
  selectedIds,
  onLassoSelect,
}: {
  data: GraphData;
  onSelect: (node: GraphNode | null) => void;
  highlight?: string;
  lassoMode?: boolean;
  selectedIds?: Set<string>;
  onLassoSelect?: (ids: string[]) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const [hoverEdge, setHoverEdge] = useState<GraphEdge | null>(null);
  const [rect, setRect] = useState<Rect | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.nodes.length === 0) return;
    const graph = new Graph({ multi: false, type: "undirected" });
    graphRef.current = graph;

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
      if (
        graph.hasNode(e.source) &&
        graph.hasNode(e.target) &&
        !graph.hasEdge(e.source, e.target)
      ) {
        graph.addEdge(e.source, e.target, {
          size: 0.5 + e.weight * 4,
          color: "#2a3142",
          weight: e.weight,
          edge: e,
        });
      }
    });

    forceAtlas2.assign(graph, {
      iterations: 200,
      settings: { gravity: 1, scalingRatio: 12 },
    });

    const renderer = new Sigma(graph, containerRef.current, {
      labelColor: { color: "#8b93a7" },
      labelSize: 11,
      defaultEdgeColor: "#2a3142",
      renderEdgeLabels: false,
    });

    renderer.on("clickNode", ({ node }) =>
      onSelect(graph.getNodeAttribute(node, "node")),
    );
    renderer.on("clickStage", () => onSelect(null));
    renderer.on("enterEdge", ({ edge }) =>
      setHoverEdge(graph.getEdgeAttribute(edge, "edge")),
    );
    renderer.on("leaveEdge", () => setHoverEdge(null));

    rendererRef.current = renderer;
    return () => {
      renderer.kill();
      rendererRef.current = null;
      graphRef.current = null;
    };
  }, [data, onSelect]);

  // Highlight nodes matching the search term and any lasso-selected nodes; dim the rest.
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const term = highlight.trim().toLowerCase();
    const sel = selectedIds && selectedIds.size > 0 ? selectedIds : null;
    renderer.setSetting("nodeReducer", (node, attrs) => {
      if (sel) {
        return sel.has(node)
          ? { ...attrs, zIndex: 2, highlighted: true, color: "#f0b429" }
          : { ...attrs, color: "#2a3142", label: "", zIndex: 0 };
      }
      if (!term) return attrs;
      const match = String(attrs.label || "")
        .toLowerCase()
        .includes(term);
      return match
        ? { ...attrs, zIndex: 1, highlighted: true }
        : { ...attrs, color: "#2a3142", label: "", zIndex: 0 };
    });
    renderer.refresh();
  }, [highlight, selectedIds]);

  // Lasso: drag a rectangle over the canvas to select the nodes inside it.
  function lassoStart(e: React.MouseEvent) {
    if (!lassoMode || !containerRef.current) return;
    const r = containerRef.current.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    dragRef.current = { x, y };
    setRect({ x0: x, y0: y, x1: x, y1: y });
  }

  function lassoMove(e: React.MouseEvent) {
    if (!dragRef.current || !containerRef.current) return;
    const r = containerRef.current.getBoundingClientRect();
    setRect({
      x0: dragRef.current.x,
      y0: dragRef.current.y,
      x1: e.clientX - r.left,
      y1: e.clientY - r.top,
    });
  }

  function lassoEnd() {
    const renderer = rendererRef.current;
    const graph = graphRef.current;
    if (!dragRef.current || !rect || !renderer || !graph) {
      dragRef.current = null;
      setRect(null);
      return;
    }
    const xMin = Math.min(rect.x0, rect.x1);
    const xMax = Math.max(rect.x0, rect.x1);
    const yMin = Math.min(rect.y0, rect.y1);
    const yMax = Math.max(rect.y0, rect.y1);
    const ids: string[] = [];
    if (xMax - xMin > 3 && yMax - yMin > 3) {
      graph.forEachNode((node) => {
        const p = renderer.graphToViewport({
          x: graph.getNodeAttribute(node, "x") as number,
          y: graph.getNodeAttribute(node, "y") as number,
        });
        if (p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax)
          ids.push(node);
      });
    }
    dragRef.current = null;
    setRect(null);
    if (ids.length > 0) onLassoSelect?.(ids);
  }

  return (
    <div className="relative h-[70vh] w-full overflow-hidden rounded-lg border border-border bg-panel">
      <div ref={containerRef} className="h-full w-full" />
      {lassoMode && (
        <div
          className="absolute inset-0 z-10 cursor-crosshair"
          onMouseDown={lassoStart}
          onMouseMove={lassoMove}
          onMouseUp={lassoEnd}
          onMouseLeave={lassoEnd}
        >
          {rect && (
            <div
              className="absolute border border-accent bg-accent/10"
              style={{
                left: Math.min(rect.x0, rect.x1),
                top: Math.min(rect.y0, rect.y1),
                width: Math.abs(rect.x1 - rect.x0),
                height: Math.abs(rect.y1 - rect.y0),
              }}
            />
          )}
        </div>
      )}
      {data.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">
          No graph yet. Ingest papers to populate it.
        </div>
      )}
      {hoverEdge && (
        <div className="absolute bottom-3 left-3 card max-w-xs text-xs">
          <div className="mb-1 font-semibold text-white">Edge anatomy</div>
          <div className="mb-1 text-muted">
            weight {hoverEdge.weight.toFixed(3)}
          </div>
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
