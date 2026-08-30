"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { CoverageProbe, CoverageReport } from "@/lib/types";

const FACETS = ["method", "dataset", "concept"];

const SOURCE_LABEL: Record<string, string> = {
  research_question: "asked by a paper",
  open_problem: "open problem",
  facet_cross: "nobody asked",
};

const SOURCE_STYLE: Record<string, string> = {
  research_question: "border-accent/50 text-accent",
  open_problem: "border-warn/50 text-warn",
  facet_cross: "border-accent2/50 text-accent2",
};

const STATE_STYLE: Record<string, string> = {
  covered: "border-good/50 text-good",
  partial: "border-warn/50 text-warn",
  uncovered: "border-bad/50 text-bad",
};

export default function CoveragePage() {
  const [rowFacet, setRowFacet] = useState("method");
  const [colFacet, setColFacet] = useState("dataset");
  const [useGlobal, setUseGlobal] = useState(false);
  const [report, setReport] = useState<CoverageReport>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const run = useCallback(async () => {
    setLoading(true);
    setError(undefined);
    try {
      setReport(await api.coverage(rowFacet, colFacet, 48, 10, useGlobal));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [rowFacet, colFacet, useGlobal]);

  useEffect(() => {
    run();
  }, [run]);

  const dropped = Object.entries(report?.dropped_by_cap ?? {});

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-3xl">
          <h1 className="text-xl font-semibold text-white">
            Question coverage
          </h1>
          <p className="text-sm text-muted">
            You cannot enumerate what a field does not know it does not know.
            You can probe it. Lattice asks the corpus questions it ought to be
            able to answer and reports the ones it cannot: the corpus&apos;s own
            research questions, its open problems, and crossings nobody asked
            about. Retrieval-only, so nothing here is invented.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <Facet
            value={rowFacet}
            onChange={setRowFacet}
            exclude={colFacet}
            label="rows"
          />
          <span className="text-muted">x</span>
          <Facet
            value={colFacet}
            onChange={setColFacet}
            exclude={rowFacet}
            label="cols"
          />
          <label
            className="flex items-center gap-1 text-xs text-muted"
            title="Query OpenAlex for the Empty vs Blind-spot signal"
          >
            <input
              type="checkbox"
              checked={useGlobal}
              onChange={(e) => setUseGlobal(e.target.checked)}
            />
            global signal
          </label>
          <button className="btn-accent" onClick={run} disabled={loading}>
            {loading ? "Probing..." : "Re-probe"}
          </button>
        </div>
      </header>

      {error && (
        <div className="card border-bad text-bad">
          Backend unreachable: {error}
        </div>
      )}

      {report && report.summary.probe_count === 0 && !error && (
        <div className="card text-sm text-muted">
          Nothing to probe yet. Ingest papers so the corpus has questions, open
          problems, and facet crossings to be tested against.
        </div>
      )}

      {report && report.summary.probe_count > 0 && (
        <>
          <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat
              label="coverage index"
              value={report.summary.coverage_index.toFixed(2)}
              hint="mean coverage across every probe"
            />
            <Stat
              label="blind-spot ratio"
              value={`${Math.round(report.summary.blind_spot_ratio * 100)}%`}
              hint="probes the corpus cannot answer at all"
            />
            <Stat
              label="probes run"
              value={String(report.summary.probe_count)}
              hint={`${report.summary.by_source.facet_cross ?? 0} nobody asked`}
            />
            <Stat
              label="covered / partial"
              value={`${report.summary.by_state.covered ?? 0} / ${
                report.summary.by_state.partial ?? 0
              }`}
              hint="grounded in the corpus text"
            />
          </section>

          <section className="space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-white">
                Blind spots, highest pressure first
              </h2>
              <p className="text-xs text-muted">
                Pressure = how badly the probe went unanswered, weighted by how
                strong a blind-spot signal its origin is and how heavily the
                corpus leans on it. These are proxies, honestly labeled, not
                proof of an unknown unknown.
              </p>
            </div>
            {report.blind_spots.map((probe, i) => (
              <BlindSpot
                key={`${probe.text}-${i}`}
                probe={probe}
                rowFacet={report.row_facet}
                colFacet={report.col_facet}
              />
            ))}
            {report.blind_spots.length === 0 && (
              <p className="card text-sm text-muted">
                Every probe was answered. That is a well-covered corpus, or a
                probe bank that is too easy.
              </p>
            )}
          </section>

          <section className="card space-y-2">
            <h2 className="text-sm font-semibold text-white">
              Every probe and its anatomy
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-muted">
                  <tr>
                    <th className="py-1 pr-3 font-medium">question</th>
                    <th className="py-1 pr-3 font-medium">origin</th>
                    <th className="py-1 pr-3 font-medium">state</th>
                    <th className="py-1 pr-3 font-medium">retrieval</th>
                    <th className="py-1 pr-3 font-medium">support</th>
                    <th className="py-1 pr-3 font-medium">grounding</th>
                    <th className="py-1 font-medium">coverage</th>
                  </tr>
                </thead>
                <tbody>
                  {report.probes.map((probe, i) => (
                    <tr
                      key={`${probe.text}-${i}`}
                      className="border-t border-border align-top"
                    >
                      <td className="max-w-md py-1.5 pr-3 text-ink">
                        {probe.text}
                      </td>
                      <td className="py-1.5 pr-3">
                        <span
                          className={`chip border ${SOURCE_STYLE[probe.source] ?? ""}`}
                        >
                          {SOURCE_LABEL[probe.source] ?? probe.source}
                        </span>
                      </td>
                      <td className="py-1.5 pr-3">
                        <span
                          className={`chip border ${STATE_STYLE[probe.state] ?? ""}`}
                        >
                          {probe.state}
                        </span>
                      </td>
                      <td className="py-1.5 pr-3">
                        <Bar value={probe.components.retrieval} />
                      </td>
                      <td className="py-1.5 pr-3">
                        <Bar value={probe.components.support} />
                      </td>
                      <td className="py-1.5 pr-3">
                        <Bar value={probe.components.grounding} />
                      </td>
                      <td className="py-1.5 font-mono text-ink">
                        {probe.coverage.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {dropped.length > 0 && (
              <p className="text-xs text-muted">
                Probe bank capped at {report.summary.probe_count}. Not run:{" "}
                {dropped.map(([src, n]) => `${n} ${src}`).join(", ")}. Raise the
                cap with the <code>limit</code> parameter.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}

function BlindSpot({
  probe,
  rowFacet,
  colFacet,
}: {
  probe: CoverageProbe;
  rowFacet: string;
  colFacet: string;
}) {
  return (
    <article className="card space-y-2">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <p className="text-sm text-ink">{probe.text}</p>
        <div className="flex shrink-0 items-center gap-1">
          <span className={`chip border ${SOURCE_STYLE[probe.source] ?? ""}`}>
            {SOURCE_LABEL[probe.source] ?? probe.source}
          </span>
          <span className={`chip border ${STATE_STYLE[probe.state] ?? ""}`}>
            {probe.state}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="w-16 shrink-0 text-xs text-muted">pressure</span>
        <Bar value={probe.pressure} tone="bad" />
        <span className="w-10 shrink-0 text-right font-mono text-xs text-muted">
          {probe.pressure.toFixed(2)}
        </span>
      </div>

      {probe.missing_terms.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-xs text-muted">no evidence mentions</span>
          {probe.missing_terms.slice(0, 8).map((term) => (
            <span key={term} className="chip border border-bad/40 text-bad">
              {term}
            </span>
          ))}
        </div>
      )}

      {probe.supporting_papers.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 text-xs text-muted">
          <span>closest evidence</span>
          {probe.supporting_papers.slice(0, 4).map((id) => (
            <Link
              key={id}
              href={`/papers/${encodeURIComponent(id)}`}
              className="chip hover:border-accent hover:text-accent"
            >
              {probe.best_evidence?.paper_id === id && probe.best_evidence.title
                ? probe.best_evidence.title
                : id.slice(0, 8)}
            </Link>
          ))}
        </div>
      )}

      {probe.facet_cell && (
        <Link
          href={`/opportunities?row=${encodeURIComponent(probe.facet_cell[0])}&col=${encodeURIComponent(
            probe.facet_cell[1],
          )}&row_facet=${rowFacet}&col_facet=${colFacet}`}
          className="inline-block text-xs text-accent hover:underline"
        >
          turn {probe.facet_cell[0]} x {probe.facet_cell[1]} into a proposal
        </Link>
      )}
    </article>
  );
}

function Bar({ value, tone = "accent" }: { value: number; tone?: string }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div
      className="h-1.5 w-full min-w-16 rounded-full bg-panel2"
      title={`${pct}%`}
      role="img"
      aria-label={`${pct} percent`}
    >
      <div
        className={`h-1.5 rounded-full ${tone === "bad" ? "bg-bad" : "bg-accent"}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="card">
      <p className="text-xs text-muted">{label}</p>
      <p className="font-mono text-xl text-white">{value}</p>
      <p className="text-[11px] text-muted">{hint}</p>
    </div>
  );
}

function Facet({
  value,
  onChange,
  exclude,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  exclude: string;
  label: string;
}) {
  return (
    <select
      aria-label={label}
      className="input w-32"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {FACETS.filter((facet) => facet !== exclude).map((facet) => (
        <option key={facet} value={facet}>
          {facet}
        </option>
      ))}
    </select>
  );
}
