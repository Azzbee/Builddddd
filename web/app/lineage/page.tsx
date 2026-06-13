"use client";

import { useState } from "react";
import { api } from "@/lib/api";

interface LineageData {
  method: string;
  nodes: { id: string; title: string; year: number | null }[];
  timeline: Record<string, string[]>;
}

export default function LineagePage() {
  const [method, setMethod] = useState("lstm");
  const [data, setData] = useState<LineageData>();
  const [error, setError] = useState<string>();

  async function load() {
    setError(undefined);
    try {
      setData(await api.lineage(method));
    } catch (e) {
      setError(String(e));
    }
  }

  const titleById = new Map((data?.nodes ?? []).map((n) => [n.id, n.title]));

  return (
    <div className="space-y-4">
      <header className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">Lineage</h1>
          <p className="text-sm text-muted">How a method family evolved over time.</p>
        </div>
        <div className="flex gap-2">
          <input
            className="input w-40"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && load()}
            placeholder="lstm"
          />
          <button className="btn-accent" onClick={load}>
            Trace
          </button>
        </div>
      </header>

      {error && <div className="card border-bad text-bad">{error}</div>}

      {data && (
        <div className="space-y-3">
          {Object.entries(data.timeline).length === 0 ? (
            <p className="text-sm text-muted">No papers use this method.</p>
          ) : (
            <div className="relative space-y-4 border-l border-border pl-6">
              {Object.entries(data.timeline)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([year, ids]) => (
                  <div key={year} className="relative">
                    <span className="absolute -left-[31px] top-1 h-3 w-3 rounded-full bg-accent" />
                    <div className="text-xs font-semibold text-accent">{year}</div>
                    <ul className="mt-1 space-y-1">
                      {ids.map((id) => (
                        <li key={id} className="text-sm text-ink">
                          {titleById.get(id) ?? id}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
