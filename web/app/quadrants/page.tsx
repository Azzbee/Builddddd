"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Row = Record<string, unknown>;

export default function QuadrantsPage() {
  const [data, setData] = useState<{
    known_knowns: Row[];
    known_unknowns: Row[];
    unknown_knowns: Row[];
  }>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    api
      .quadrants()
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-semibold text-white">
          Epistemic quadrants
        </h1>
        <p className="text-sm text-muted">
          Each quadrant is a computable query, not a vibe. Unknown unknowns are
          surfaced as proxies in the gap matrix.
        </p>
      </header>

      {error && <div className="card border-bad text-bad">{error}</div>}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Quadrant
          title="Known knowns"
          hint="independently supported, non-contradicted"
        >
          {(data?.known_knowns ?? []).map((f, i) => (
            <li key={i} className="text-sm">
              <span className="text-ink">{String(f.claim)}</span>
              <span className="ml-1 text-xs text-muted">
                ({String(f.independent_supports)} supports
                {f.triangulated ? ", triangulated" : ""})
              </span>
            </li>
          ))}
        </Quadrant>
        <Quadrant title="Known unknowns" hint="the field's own open problems">
          {(data?.known_unknowns ?? []).map((p, i) => (
            <li key={i} className="text-sm">
              <span className="text-ink">{String(p.problem)}</span>
              <span className="ml-1 text-xs text-muted">
                (raised {String(p.frequency)}x, span {String(p.span_years)}y)
              </span>
            </li>
          ))}
        </Quadrant>
        <Quadrant title="Unknown knowns" hint="the answer exists next door">
          {(data?.unknown_knowns ?? []).map((t, i) => (
            <li key={i} className="text-sm">
              <span className="text-ink">{String(t.method)}</span>
              <span className="ml-1 text-xs text-muted">
                (sim {Number(t.similarity).toFixed(2)}, distance{" "}
                {String(t.graph_distance)})
              </span>
            </li>
          ))}
        </Quadrant>
      </div>
    </div>
  );
}

function Quadrant({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <section className="card">
      <h2 className="text-sm font-semibold text-white">{title}</h2>
      <p className="mb-2 text-xs text-muted">{hint}</p>
      {items.length === 0 ? (
        <p className="text-sm text-muted">nothing yet</p>
      ) : (
        <ul className="list-disc space-y-1 pl-5">{children}</ul>
      )}
    </section>
  );
}
