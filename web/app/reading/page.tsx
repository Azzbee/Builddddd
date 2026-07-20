"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";

interface QueueItem {
  paper_id: string;
  title: string;
  gain: number;
  neighborhood_value: number;
  method_novelty: number;
  novel_methods: string[];
}

export default function ReadingPage() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [error, setError] = useState<string>();

  useEffect(() => {
    api
      .readingQueue()
      .then((r) => setQueue(r.queue as unknown as QueueItem[]))
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-semibold text-white">Reading queue</h1>
        <p className="text-sm text-muted">
          Ranked by expected information gain: how central the paper is to your
          graph times how novel its methods are.
        </p>
      </header>

      {error && <div className="card border-bad text-bad">{error}</div>}

      {queue.length === 0 ? (
        <p className="text-sm text-muted">Nothing to rank yet.</p>
      ) : (
        <ol className="space-y-2">
          {queue.map((q, i) => (
            <li key={q.paper_id} className="card flex items-center gap-3">
              <span className="font-mono text-sm text-muted">{i + 1}</span>
              <div className="flex-1">
                <Link
                  href={`/papers/${q.paper_id}`}
                  className="text-sm text-ink hover:text-white"
                >
                  {q.title}
                </Link>
                <div className="mt-1 flex flex-wrap gap-1">
                  <span className="chip">gain {q.gain.toFixed(2)}</span>
                  <span className="chip">
                    centrality {q.neighborhood_value.toFixed(2)}
                  </span>
                  <span className="chip">
                    novelty {q.method_novelty.toFixed(2)}
                  </span>
                  {q.novel_methods.slice(0, 3).map((m) => (
                    <span key={m} className="chip border-accent text-accent">
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
