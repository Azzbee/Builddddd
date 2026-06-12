"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { PaperSummary } from "@/lib/types";

export default function PapersPage() {
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [q, setQ] = useState("");
  const [error, setError] = useState<string>();

  useEffect(() => {
    api.listPapers().then(setPapers).catch((e) => setError(String(e)));
  }, []);

  const filtered = useMemo(
    () =>
      papers.filter(
        (p) =>
          p.title.toLowerCase().includes(q.toLowerCase()) ||
          p.authors.join(" ").toLowerCase().includes(q.toLowerCase()),
      ),
    [papers, q],
  );

  return (
    <div className="space-y-4">
      <header className="flex items-center justify-between gap-3">
        <h1 className="text-xl font-semibold text-white">Papers ({papers.length})</h1>
        <input
          className="input max-w-xs"
          placeholder="Filter by title or author"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </header>
      {error && <div className="card border-bad text-bad">Backend unreachable: {error}</div>}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {filtered.map((p) => (
          <Link key={p.paper_id} href={`/papers/${p.paper_id}`} className="card hover:border-accent">
            <div className="flex items-start justify-between gap-2">
              <h2 className="text-sm font-medium text-white">{p.title}</h2>
              {p.needs_review && <span className="chip border-warn text-warn">review</span>}
            </div>
            <p className="mt-1 truncate text-xs text-muted">{p.authors.join(", ") || "unknown"}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              <span className="chip">{p.year ?? "?"}</span>
              <span className="chip">{p.paper_type}</span>
              <span className="chip">conf {p.confidence.toFixed(2)}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
