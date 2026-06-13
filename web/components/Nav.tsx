"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/graph", label: "Graph" },
  { href: "/papers", label: "Papers" },
  { href: "/chat", label: "Chat" },
  { href: "/ingest", label: "Ingest" },
  { href: "/landscape", label: "Landscape" },
  { href: "/quadrants", label: "Quadrants" },
  { href: "/contradictions", label: "Contradictions" },
  { href: "/lineage", label: "Lineage" },
  { href: "/reading", label: "Reading" },
  { href: "/related", label: "Related work" },
  { href: "/digest", label: "Digest" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="flex items-center gap-1 border-b border-border bg-panel px-4 py-2.5">
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
    </nav>
  );
}
