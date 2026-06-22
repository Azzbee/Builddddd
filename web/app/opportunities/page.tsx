"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { ProposalEvidence, ResearchProposal } from "@/lib/types";

const FACETS = ["method", "dataset", "concept"];

const STATE_STYLE: Record<string, string> = {
  gap: "border-good text-good",
  blind_spot: "border-warn text-warn",
  occupied: "border-border text-muted",
};

export default function OpportunitiesPage() {
  const [rowFacet, setRowFacet] = useState("method");
  const [colFacet, setColFacet] = useState("dataset");
  const [useGlobal, setUseGlobal] = useState(false);
  const [proposals, setProposals] = useState<ResearchProposal[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  async function run() {
    setLoading(true);
    setError(undefined);
    try {
      const r = await api.opportunities(rowFacet, colFacet, 6, useGlobal);
      setProposals(r.proposals);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">Research opportunities</h1>
          <p className="text-sm text-muted">
            The corpus&apos;s highest-pressure gaps, each turned into a grounded proposal:
            what to try, the building blocks already in your library, why now, and the risks.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <Facet value={rowFacet} onChange={setRowFacet} exclude={colFacet} label="rows" />
          <span className="text-muted">x</span>
          <Facet value={colFacet} onChange={setColFacet} exclude={rowFacet} label="cols" />
          <label className="flex items-center gap-1 text-xs text-muted" title="Query OpenAlex for the Empty vs Blind-spot signal">
            <input
              type="checkbox"
              checked={useGlobal}
              onChange={(e) => setUseGlobal(e.target.checked)}
            />
            global signal
          </label>
          <button className="btn-accent" onClick={run} disabled={loading}>
            {loading ? "Thinking..." : "Find gaps"}
          </button>
        </div>
      </header>

      {error && <div className="card border-bad text-bad">Backend unreachable: {error}</div>}
      {!loading && proposals.length === 0 && !error && (
        <div className="card text-sm text-muted">
          No empty cells for this facet pair — try another combination, or ingest more papers.
        </div>
      )}

      <div className="space-y-4">
        {proposals.map((p, i) => (
          <ProposalCard key={`${p.row}-${p.col}-${i}`} p={p} />
        ))}
      </div>
    </div>
  );
}

function ProposalCard({ p }: { p: ResearchProposal }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    await navigator.clipboard.writeText(p.markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }
  return (
    <article className="card space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-white">
            {p.row} <span className="text-muted">x</span> {p.col}
          </h2>
          <p className="mt-1 text-sm text-ink">{p.thesis}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`chip border ${STATE_STYLE[p.state] ?? ""}`}>{p.state.replace("_", " ")}</span>
          <span className="text-xs text-muted">confidence {p.confidence.toFixed(2)}</span>
          <button className="text-xs text-accent hover:underline" onClick={copy}>
            {copied ? "copied" : "copy brief"}
          </button>
        </div>
      </div>

      <p className="text-xs italic text-muted">{p.novelty}</p>

      <div className="flex flex-wrap gap-1">
        <span className="chip">why now: {p.why_now}</span>
        {p.global_count > 0 && <span className="chip">~{p.global_count} papers worldwide</span>}
      </div>

      {p.method_to_borrow && (
        <Block label="Method to borrow">
          <PaperLink e={p.method_to_borrow} />
        </Block>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {p.row_track_record.length > 0 && (
          <Block label={`Track record · ${p.row}`}>
            <EvidenceList items={p.row_track_record} />
          </Block>
        )}
        {p.col_track_record.length > 0 && (
          <Block label={`Track record · ${p.col}`}>
            <EvidenceList items={p.col_track_record} />
          </Block>
        )}
      </div>

      {p.recommended_baselines.length > 0 && (
        <Block label="Baselines to beat">
          <div className="flex flex-wrap gap-1">
            {p.recommended_baselines.map((b) => (
              <span key={b} className="chip">{b}</span>
            ))}
          </div>
        </Block>
      )}

      {p.open_problems.length > 0 && (
        <Block label="Open problems this addresses">
          <ul className="list-disc space-y-0.5 pl-4 text-xs text-ink">
            {p.open_problems.map((o, i) => <li key={i}>{o}</li>)}
          </ul>
        </Block>
      )}

      {p.risks.length > 0 && (
        <Block label="Risks">
          <ul className="list-disc space-y-0.5 pl-4 text-xs text-muted">
            {p.risks.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </Block>
      )}
    </article>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">{label}</div>
      {children}
    </div>
  );
}

function EvidenceList({ items }: { items: ProposalEvidence[] }) {
  return (
    <ul className="space-y-0.5 text-xs">
      {items.map((e) => (
        <li key={e.paper_id}>
          <PaperLink e={e} />
        </li>
      ))}
    </ul>
  );
}

function PaperLink({ e }: { e: ProposalEvidence }) {
  return (
    <span>
      <Link href={`/papers/${e.paper_id}`} className="text-accent hover:underline">
        {e.title}
      </Link>{" "}
      <span className="text-muted">
        ({e.year ?? "?"}) — {e.note}
      </span>
    </span>
  );
}

function Facet({
  value,
  onChange,
  exclude,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  exclude: string;
  label: string;
}) {
  return (
    <label className="flex items-center gap-1 text-xs text-muted">
      {label}
      <select
        className="rounded-md border border-border bg-panel2 px-2 py-1 text-sm text-ink outline-none focus:border-accent"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {FACETS.filter((f) => f !== exclude).map((f) => (
          <option key={f} value={f}>
            {f}
          </option>
        ))}
      </select>
    </label>
  );
}
