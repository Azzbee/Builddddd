"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { PaperSummary } from "@/lib/types";

export default function Overview() {
  const [stats, setStats] = useState<{ papers: number; edges: number; communities: number }>();
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [error, setError] = useState<string>();

  useEffect(() => {
    Promise.all([api.graphStats(), api.listPapers()])
      .then(([s, p]) => {
        setStats(s);
        setPapers(p);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const needsReview = papers.filter((p) => p.needs_review).length;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-white">Overview</h1>
        <p className="text-sm text-muted">A living map of your research field.</p>
      </header>

      {error && <div className="card border-bad text-bad">Backend unreachable: {error}</div>}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Papers" value={stats?.papers ?? papers.length} />
        <Stat label="Relationships" value={stats?.edges ?? 0} />
        <Stat label="Communities" value={stats?.communities ?? 0} />
        <Stat label="Needs review" value={needsReview} warn={needsReview > 0} />
      </div>

      <section className="card">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Recent papers</h2>
          <Link href="/papers" className="text-xs text-accent hover:underline">
            view all
          </Link>
        </div>
        {papers.length === 0 ? (
          <Empty />
        ) : (
          <ul className="divide-y divide-border">
            {papers.slice(0, 8).map((p) => (
              <li key={p.paper_id} className="py-2">
                <Link
                  href={`/papers/${p.paper_id}`}
                  className="flex items-center justify-between gap-3 hover:text-white"
                >
                  <span className="truncate text-sm">{p.title}</span>
                  <span className="shrink-0 text-xs text-muted">
                    {p.year ?? "?"} · {p.paper_type}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, warn = false }: { label: string; value: number; warn?: boolean }) {
  return (
    <div className="card">
      <div className={`text-2xl font-semibold ${warn ? "text-warn" : "text-white"}`}>{value}</div>
      <div className="text-xs text-muted">{label}</div>
    </div>
  );
}

function Empty() {
  return (
    <p className="text-sm text-muted">
      No papers yet. Head to <Link href="/ingest" className="text-accent">Ingest</Link> to add some.
    </p>
  );
}
