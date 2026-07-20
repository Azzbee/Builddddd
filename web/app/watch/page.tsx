"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Candidate {
  arxiv_id: string;
  title: string;
  similarity: number;
  nearest_paper_id: string | null;
  pdf_url: string | null;
}

export default function WatchPage() {
  const [queue, setQueue] = useState<Candidate[]>([]);
  const [error, setError] = useState<string>();

  const refresh = useCallback(() => {
    api
      .watchQueue()
      .then((q) => setQueue(q as unknown as Candidate[]))
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function decide(arxiv_id: string, approve: boolean) {
    await api.approveWatch(arxiv_id, approve);
    refresh();
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-semibold text-white">Watch queue</h1>
        <p className="text-sm text-muted">
          arXiv candidates similar to your corpus. Approve to ingest, reject to
          dismiss. Nothing is ingested automatically.
        </p>
      </header>

      {error && <div className="card border-bad text-bad">{error}</div>}

      {queue.length === 0 ? (
        <p className="text-sm text-muted">
          Nothing pending. The watcher queues matches on its schedule (every
          6h).
        </p>
      ) : (
        <ul className="space-y-2">
          {queue.map((c) => (
            <li
              key={c.arxiv_id}
              className="card flex items-center justify-between gap-3"
            >
              <div className="min-w-0">
                <div className="truncate text-sm text-ink">{c.title}</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  <span className="chip">arXiv:{c.arxiv_id}</span>
                  <span className="chip">
                    similarity {Number(c.similarity).toFixed(2)}
                  </span>
                  {c.pdf_url && (
                    <a
                      className="chip text-accent"
                      href={c.pdf_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      pdf
                    </a>
                  )}
                </div>
              </div>
              <div className="flex shrink-0 gap-2">
                <button
                  className="btn-accent"
                  onClick={() => decide(c.arxiv_id, true)}
                >
                  Approve
                </button>
                <button
                  className="btn"
                  onClick={() => decide(c.arxiv_id, false)}
                >
                  Reject
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
