"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { getWorkspace, setWorkspace } from "@/lib/workspace";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/graph", label: "Graph" },
  { href: "/papers", label: "Papers" },
  { href: "/chat", label: "Chat" },
  { href: "/ingest", label: "Ingest" },
  { href: "/landscape", label: "Landscape" },
  { href: "/opportunities", label: "Opportunities" },
  { href: "/quadrants", label: "Quadrants" },
  { href: "/contradictions", label: "Contradictions" },
  { href: "/lineage", label: "Lineage" },
  { href: "/reading", label: "Reading" },
  { href: "/related", label: "Related work" },
  { href: "/watch", label: "Watch" },
  { href: "/digest", label: "Digest" },
];

export function Nav() {
  const path = usePathname();
  const [ws, setWs] = useState("default");

  useEffect(() => setWs(getWorkspace()), []);

  function commit(value: string) {
    const v = value.trim() || "default";
    setWs(v);
    setWorkspace(v);
    window.location.reload(); // re-fetch all views for the new corpus
  }

  return (
    <nav className="flex flex-wrap items-center gap-1 border-b border-border bg-panel px-4 py-2.5">
      <Link href="/" className="mr-4 font-mono text-sm font-semibold tracking-tight text-white">
        lattice<span className="text-accent">.</span>
      </Link>
      {LINKS.map((l) => {
        const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`rounded-md px-3 py-1.5 text-sm transition ${
              active ? "bg-panel2 text-white" : "text-muted hover:text-ink"
            }`}
          >
            {l.label}
          </Link>
        );
      })}
      <label className="ml-auto flex items-center gap-1 text-xs text-muted" title="Corpus / workspace">
        corpus
        <input
          key={ws}
          aria-label="workspace"
          defaultValue={ws}
          onBlur={(e) => e.target.value.trim() !== ws && commit(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && commit((e.target as HTMLInputElement).value)}
          className="w-28 rounded-md border border-border bg-panel2 px-2 py-1 text-xs text-ink outline-none focus:border-accent"
        />
      </label>
    </nav>
  );
}
