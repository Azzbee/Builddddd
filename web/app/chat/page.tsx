"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { streamQuery } from "@/lib/api";
import type { Citation } from "@/lib/types";

interface Trace {
  type: string;
  label: string;
}

interface Turn {
  question: string;
  answer: string;
  citations: Citation[];
  confidence: number;
  abstained: boolean;
  trace: Trace[];
  done: boolean;
}

export default function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const cancelRef = useRef<(() => void) | undefined>(undefined);

  function ask() {
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    const idx = turns.length;
    setTurns((t) => [
      ...t,
      { question, answer: "", citations: [], confidence: 0, abstained: false, trace: [], done: false },
    ]);

    cancelRef.current = streamQuery(question, (type, data) => {
      setTurns((prev) => {
        const next = [...prev];
        const turn = { ...next[idx] };
        if (type === "status") {
          turn.trace = [...turn.trace, { type, label: `classified: ${data.query_class}` }];
        } else if (type === "tool_call") {
          turn.trace = [...turn.trace, { type, label: `tool: ${data.name}` }];
        } else if (type === "final") {
          turn.answer = String(data.answer ?? "");
          turn.citations = (data.citations as Citation[]) ?? [];
          turn.confidence = Number(data.confidence ?? 0);
          turn.abstained = Boolean(data.abstained);
          turn.done = true;
          setBusy(false);
        }
        next[idx] = turn;
        return next;
      });
    });
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-semibold text-white">Chat</h1>
        <p className="text-sm text-muted">
          Ask about the corpus. Answers are grounded with citations; the agent says
          &quot;I don&apos;t know&quot; when evidence is thin.
        </p>
      </header>

      <div className="space-y-4">
        {turns.map((t, i) => (
          <div key={i} className="space-y-2">
            <div className="text-sm font-medium text-accent">{t.question}</div>
            {t.trace.length > 0 && (
              <details className="text-xs text-muted">
                <summary className="cursor-pointer">how I got this ({t.trace.length} steps)</summary>
                <ul className="mt-1 space-y-0.5 pl-4">
                  {t.trace.map((s, j) => (
                    <li key={j}>· {s.label}</li>
                  ))}
                </ul>
              </details>
            )}
            <div className="card">
              {t.done ? (
                <>
                  {t.abstained && <div className="mb-1 chip border-warn text-warn">low confidence</div>}
                  <p className="whitespace-pre-wrap text-sm text-ink">{t.answer}</p>
                  {t.citations.length > 0 && (
                    <ol className="mt-3 space-y-1 border-t border-border pt-2 text-xs text-muted">
                      {t.citations.map((c) => (
                        <li key={c.marker}>
                          [{c.marker}]{" "}
                          <Link href={`/papers/${c.paper_id}`} className="text-accent hover:underline">
                            {c.title || c.paper_id}
                          </Link>
                          {c.evidence_location ? ` · ${c.evidence_location}` : ""}
                        </li>
                      ))}
                    </ol>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted">thinking...</p>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          className="input"
          placeholder="Which methods have been tried for X, and where do they disagree?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button className="btn-accent" onClick={ask} disabled={busy}>
          Ask
        </button>
      </div>
    </div>
  );
}
