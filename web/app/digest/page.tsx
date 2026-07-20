"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Mover {
  concept: string;
  composite: number;
  maturity: string;
  velocity: number;
  acceleration: number;
}

export default function DigestPage() {
  const [movers, setMovers] = useState<Mover[]>([]);
  const [digest, setDigest] = useState<string>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    api
      .momentum()
      .then((m) => setMovers((m.movers as unknown as Mover[]) ?? []))
      .catch((e) => setError(String(e)));
  }, []);

  async function generate() {
    try {
      setDigest((await api.generateDigest()).markdown);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">Digest: movers</h1>
          <p className="text-sm text-muted">
            Concept momentum from the corpus. Momentum measures research
            activity, not truth: pair it with the contested and saturated states
            before believing a trend.
          </p>
        </div>
        <button className="btn-accent" onClick={generate}>
          Generate digest
        </button>
      </header>

      {digest && (
        <section className="card">
          <pre className="whitespace-pre-wrap font-sans text-sm text-ink">
            {digest}
          </pre>
        </section>
      )}

      {error && (
        <div className="card border-bad text-bad">
          Backend unreachable: {error}
        </div>
      )}

      {movers.length === 0 ? (
        <p className="text-sm text-muted">
          No movers yet. Ingest more papers across years.
        </p>
      ) : (
        <div className="space-y-2">
          {movers.map((m) => (
            <div key={m.concept} className="card">
              <div className="flex items-center justify-between">
                <span className="font-medium text-white">{m.concept}</span>
                <span className="chip">{m.maturity}</span>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded bg-panel2">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${Math.round(m.composite * 100)}%` }}
                />
              </div>
              <div className="mt-1 flex gap-3 text-xs text-muted">
                <span>momentum {m.composite.toFixed(2)}</span>
                <span>velocity {m.velocity.toFixed(1)}</span>
                <span>acceleration {m.acceleration.toFixed(1)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
