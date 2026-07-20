"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function RelatedWorkPage() {
  const [markdown, setMarkdown] = useState("");
  const [bibtex, setBibtex] = useState("");
  const [error, setError] = useState<string>();

  useEffect(() => {
    api
      .relatedWork()
      .then((r) => {
        setMarkdown(r.markdown);
        setBibtex(r.bibtex);
      })
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">Related work</h1>
          <p className="text-sm text-muted">
            A grounded, hedged draft grouped by graph community, with citations.
          </p>
        </div>
        <div className="flex gap-2">
          {/* API responses are file downloads, not client-routed pages. */}
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
          <a className="btn" href="/api/export/bibtex">
            Download .bib
          </a>
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
          <a className="btn" href="/api/export/obsidian">
            Obsidian .zip
          </a>
        </div>
      </header>

      {error && <div className="card border-bad text-bad">{error}</div>}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <section className="card">
          <h2 className="mb-2 text-sm font-semibold text-white">Draft</h2>
          <pre className="whitespace-pre-wrap font-sans text-sm text-ink">
            {markdown || "Ingest papers first."}
          </pre>
        </section>
        <section className="card">
          <h2 className="mb-2 text-sm font-semibold text-white">BibTeX</h2>
          <pre className="overflow-auto font-mono text-xs text-muted">
            {bibtex || "(empty)"}
          </pre>
        </section>
      </div>
    </div>
  );
}
