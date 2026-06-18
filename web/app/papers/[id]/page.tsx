"use client";

import { use, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { PaperCard, PdfMeta } from "@/lib/types";

export default function PaperPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [card, setCard] = useState<PaperCard | null>(null);
  const [error, setError] = useState<string>();
  const [initialPage, setInitialPage] = useState<number | undefined>(undefined);

  useEffect(() => {
    api.getPaper(id).then(setCard).catch((e) => setError(String(e)));
  }, [id]);

  useEffect(() => {
    // Deep-link target like ?page=8 (set by chat citations), read client-side.
    const p = Number(new URLSearchParams(window.location.search).get("page"));
    if (p > 0) setInitialPage(p);
  }, []);

  if (error) return <div className="card border-bad text-bad">Not found: {error}</div>;
  if (!card) return <div className="text-sm text-muted">Loading...</div>;

  return (
    <article className="space-y-5">
      <header>
        <h1 className="text-xl font-semibold text-white">{card.title}</h1>
        <p className="text-sm text-muted">
          {card.authors.map((a) => a.name).join(", ")} · {card.year ?? "?"}
          {card.venue ? ` · ${card.venue}` : ""}
        </p>
        <div className="mt-2 flex flex-wrap gap-1">
          <span className="chip">{card.paper_type}</span>
          <span className="chip">confidence {card.confidence.toFixed(2)}</span>
          {card.doi && <span className="chip">doi:{card.doi}</span>}
          {card.arxiv_id && <span className="chip">arXiv:{card.arxiv_id}</span>}
          {card.needs_review && <span className="chip border-warn text-warn">needs review</span>}
        </div>
      </header>

      <Section title="Problem">
        <p className="text-sm text-ink">{card.problem_statement || "n/a"}</p>
      </Section>

      <Section title="Methodology">
        <p className="text-sm text-ink">{card.methodology.approach_summary}</p>
        <Tags label="techniques" items={card.methodology.techniques} />
        <Tags label="baselines" items={card.methodology.baselines} />
      </Section>

      <Section title="Datasets">
        {card.datasets.length === 0 ? (
          <p className="text-sm text-muted">none stated</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {card.datasets.map((d, i) => (
              <li key={i}>
                <span className="text-ink">{d.name}</span>
                {d.source && <span className="text-muted"> · {d.source}</span>}
                {d.is_public != null && (
                  <span className="text-muted"> · {d.is_public ? "public" : "private"}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Key results">
        {card.key_results.length === 0 ? (
          <p className="text-sm text-muted">none extracted</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {card.key_results.map((r, i) => (
              <li key={i} className="rounded-md border border-border bg-panel2 p-2">
                <div className="text-ink">{r.claim}</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {r.metric && <span className="chip">{r.metric}{r.value ? ` ${r.value}` : ""}</span>}
                  {r.baseline_comparison && <span className="chip">vs {r.baseline_comparison}</span>}
                  {r.evidence_location && (
                    <span className="chip border-accent text-accent">{r.evidence_location}</span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Section title="Contributions">
          <Bullets items={card.contributions} />
        </Section>
        <Section title="Limitations">
          <Bullets items={card.limitations} />
        </Section>
      </div>

      <Section title="Future work">
        <Bullets items={card.future_work} />
      </Section>

      <PdfReader paperId={id} initialPage={initialPage} />
    </article>
  );
}

function PdfReader({ paperId, initialPage }: { paperId: string; initialPage?: number }) {
  const [meta, setMeta] = useState<PdfMeta | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [page, setPage] = useState<number | undefined>(initialPage);
  const [open, setOpen] = useState<boolean>(initialPage != null);

  useEffect(() => {
    api.pdfMeta(paperId).then(setMeta).catch(() => setMeta(null));
  }, [paperId]);

  // A deep-link page may resolve after mount; open the reader at it when it does.
  useEffect(() => {
    if (initialPage != null) {
      setPage(initialPage);
      setOpen(true);
    }
  }, [initialPage]);

  useEffect(() => {
    if (!open || url) return;
    let revoked: string | null = null;
    api
      .pdfObjectUrl(paperId)
      .then((u) => {
        revoked = u;
        setUrl(u);
      })
      .catch(() => setUrl(null));
    return () => {
      if (revoked) URL.revokeObjectURL(revoked);
    };
  }, [open, url, paperId]);

  if (!meta || !meta.available) return null;

  // A real, parseable PDF reports a page count; synthetic/demo bytes report 0, so
  // we surface "stored" without an embed that would just render broken.
  const readable = meta.pages > 0;

  return (
    <section className="card">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Source PDF</h2>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span>{readable ? `${meta.pages} pages` : "stored (no preview)"}</span>
          {readable && (
            <button className="btn" onClick={() => setOpen((v) => !v)}>
              {open ? "Hide" : "Open reader"}
            </button>
          )}
        </div>
      </div>
      {open &&
        readable &&
        (url ? (
          <iframe
            title="source pdf"
            src={`${url}${page ? `#page=${page}` : ""}`}
            className="h-[80vh] w-full rounded-md border border-border bg-white"
          />
        ) : (
          <p className="text-sm text-muted">Loading PDF...</p>
        ))}
      {open && readable && page && (
        <p className="mt-1 text-xs text-muted">
          Jumped to page {page}.{" "}
          <button className="text-accent hover:underline" onClick={() => setPage(undefined)}>
            show from start
          </button>
        </p>
      )}
    </section>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card">
      <h2 className="mb-2 text-sm font-semibold text-white">{title}</h2>
      {children}
    </section>
  );
}

function Tags({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1">
      <span className="text-xs text-muted">{label}:</span>
      {items.map((t) => (
        <span key={t} className="chip">
          {t}
        </span>
      ))}
    </div>
  );
}

function Bullets({ items }: { items: string[] }) {
  if (items.length === 0) return <p className="text-sm text-muted">none</p>;
  return (
    <ul className="list-disc space-y-1 pl-5 text-sm text-ink">
      {items.map((x, i) => (
        <li key={i}>{x}</li>
      ))}
    </ul>
  );
}
