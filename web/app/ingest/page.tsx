"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { IngestJob } from "@/lib/types";

const STAGES = [
  "parsing",
  "extracting",
  "enriching",
  "embedding",
  "linking",
  "done",
];

function StageBadges({ job }: { job: IngestJob }) {
  const reached =
    job.status === "succeeded" ? STAGES.length : STAGES.indexOf(job.stage) + 1;
  return (
    <div className="flex flex-wrap gap-1">
      {STAGES.map((s, i) => {
        const ok = i < reached;
        const failed = job.status === "failed" && i === reached;
        return (
          <span
            key={s}
            className={`chip ${ok ? "border-good text-good" : failed ? "border-bad text-bad" : ""}`}
          >
            {s}
          </span>
        );
      })}
    </div>
  );
}

export default function IngestPage() {
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [arxiv, setArxiv] = useState("");
  const [busy, setBusy] = useState(false);
  const [retryingJobId, setRetryingJobId] = useState<string>();
  const [msg, setMsg] = useState<string>();

  const refresh = useCallback(() => {
    api
      .jobs()
      .then(setJobs)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setMsg(undefined);
    try {
      const job = await api.ingestFile(file);
      setMsg(`Submitted ${file.name}: ${job.status}`);
      refresh();
    } catch (err) {
      setMsg(`Failed: ${String(err)}`);
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  }

  async function onRetry(job: IngestJob) {
    setRetryingJobId(job.job_id);
    setMsg(undefined);
    try {
      const updated = await api.retryJob(job.job_id);
      setMsg(`Retry submitted for ${job.source_ref}: ${updated.status}`);
      refresh();
    } catch (err) {
      setMsg(`Retry failed: ${String(err)}`);
    } finally {
      setRetryingJobId(undefined);
    }
  }

  async function onArxiv() {
    if (!arxiv.trim()) return;
    setBusy(true);
    setMsg(undefined);
    try {
      const job = await api.ingestArxiv(arxiv.trim());
      setMsg(`Queued arXiv:${arxiv}: ${job.status}`);
      setArxiv("");
      refresh();
    } catch (err) {
      setMsg(`Failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-semibold text-white">Ingest</h1>
        <p className="text-sm text-muted">
          Drop a PDF or paste an arXiv id. Watch the pipeline run.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <label className="card flex cursor-pointer flex-col items-center justify-center gap-2 border-dashed py-10 text-center hover:border-accent">
          <span className="text-sm text-ink">Click to upload a PDF</span>
          <span className="text-xs text-muted">
            parsed, extracted, enriched, embedded, linked
          </span>
          <input
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={onFile}
            disabled={busy}
          />
        </label>
        <div className="card space-y-2">
          <label className="text-sm text-ink">Ingest from arXiv</label>
          <div className="flex gap-2">
            <input
              className="input"
              placeholder="2401.01234"
              value={arxiv}
              onChange={(e) => setArxiv(e.target.value)}
            />
            <button className="btn-accent" onClick={onArxiv} disabled={busy}>
              Fetch
            </button>
          </div>
          {msg && <p className="text-xs text-muted">{msg}</p>}
        </div>
      </div>

      <section className="card">
        <h2 className="mb-3 text-sm font-semibold text-white">Jobs</h2>
        {jobs.length === 0 ? (
          <p className="text-sm text-muted">No jobs yet.</p>
        ) : (
          <ul className="space-y-3">
            {jobs.map((j) => (
              <li
                key={j.job_id}
                className="rounded-md border border-border bg-panel2 p-3"
              >
                <div className="mb-2 flex items-start justify-between gap-3 text-xs">
                  <div className="min-w-0">
                    <div className="truncate text-ink">{j.source_ref}</div>
                    <div className="font-mono text-muted">
                      {j.paper_id ?? j.job_id.slice(0, 8)} · attempts{" "}
                      {j.attempts}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span
                      className={
                        j.status === "succeeded"
                          ? "text-good"
                          : j.status === "failed"
                            ? "text-bad"
                            : "text-warn"
                      }
                    >
                      {j.status}
                      {j.error_code ? ` (${j.error_code})` : ""}
                    </span>
                    {j.retryable &&
                      (j.status === "failed" || j.status === "paused") && (
                        <button
                          className="btn py-1 text-xs"
                          onClick={() => onRetry(j)}
                          disabled={retryingJobId === j.job_id}
                        >
                          {retryingJobId === j.job_id ? "Retrying" : "Retry"}
                        </button>
                      )}
                  </div>
                </div>
                <StageBadges job={j} />
                {j.error_message && (
                  <p className="mt-2 text-xs text-bad">{j.error_message}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
