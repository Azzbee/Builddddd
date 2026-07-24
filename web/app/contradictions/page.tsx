"use client";

import { useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { ClaimRelationEdge } from "@/lib/types";

export default function ContradictionsPage() {
  const [edges, setEdges] = useState<ClaimRelationEdge[]>([]);
  const [summary, setSummary] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  async function run() {
    setBusy(true);
    setError(undefined);
    try {
      const s = await api.analyzeContradictions();
      setSummary(
        `${s.analyzed} relations, ${s.contradictions} contradictions, ${s.supports} supports`,
      );
      const list = await api.contradictions("CONTRADICTS");
      setEdges(list);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">Contradictions</h1>
          <p className="text-sm text-muted">
            Where the corpus disagrees: claims on the same concept that
            conflict.
          </p>
        </div>
        <button className="btn-accent" onClick={run} disabled={busy}>
          {busy ? "Analyzing..." : "Analyze corpus"}
        </button>
      </header>

      {error && <div className="card border-bad text-bad">{error}</div>}
      {summary && <div className="text-sm text-muted">{summary}</div>}

      {edges.length === 0 ? (
        <p className="text-sm text-muted">
          No contradictions found yet. Run the analysis after ingesting papers.
        </p>
      ) : (
        <div className="space-y-3">
          {edges.map((e, i) => (
            <div key={i} className="card border-warn/40">
              <div className="mb-2 flex items-center gap-2">
                <span className="chip border-warn text-warn">CONTRADICTS</span>
                <span className="text-xs text-muted">
                  confidence {e.confidence.toFixed(2)}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                <Side
                  text={e.source_text}
                  paper={e.source_paper}
                  evidence={e.source_evidence}
                />
                <Side
                  text={e.target_text}
                  paper={e.target_paper}
                  evidence={e.target_evidence}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Side({
  text,
  paper,
  evidence,
}: {
  text: string;
  paper: string;
  evidence: string;
}) {
  return (
    <div className="rounded-md border border-border bg-panel2 p-2 text-sm">
      <p className="text-ink">{text}</p>
      <div className="mt-1 flex items-center gap-2 text-xs text-muted">
        <Link href={`/papers/${paper}`} className="text-accent hover:underline">
          paper
        </Link>
        {evidence && <span>· {evidence}</span>}
      </div>
    </div>
  );
}
