"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { shortSha } from "@/lib/format";
import { useResource } from "@/lib/useResource";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/workspace", label: "Solver workspace" },
  { href: "/results", label: "Results" },
  { href: "/model", label: "Model" },
];

function ApiStatus() {
  const h = useResource(() => api.health());
  if (h.status === "loading") return <span className="api-status" data-state="loading">Connecting…</span>;
  if (h.status === "error")
    return (
      <span className="api-status" data-state="down" data-testid="api-status" title={h.error.message}>
        <span className="dot" /> API unavailable
      </span>
    );
  return (
    <span className="api-status" data-state="up" data-testid="api-status" title={`checkpoint sha256 ${h.data.checkpoint_sha256}`}>
      <span className="dot" /> API online · <span className="mono">{shortSha(h.data.checkpoint_sha256)}</span>
    </span>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  return (
    <div className="shell">
      <aside className="sidebar">
        <Link href="/" className="brand" aria-label="PDEFlow home">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
            <rect x="1" y="1" width="30" height="30" rx="7" fill="var(--accent)" />
            <path d="M6 20c4-8 8-8 10 0s6 8 10 0" stroke="#fff" strokeWidth="2.2" fill="none" strokeLinecap="round" />
            <path d="M6 13c4-5 8-5 10 0s6 5 10 0" stroke="#bcd3e6" strokeWidth="1.6" fill="none" strokeLinecap="round" />
          </svg>
          <span>
            <span className="brand__name">PDEFlow</span>
            <span className="brand__tag">PDE solutions as a service</span>
          </span>
        </Link>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path?.startsWith(n.href);
            return (
              <Link key={n.href} href={n.href} className={`nav__link ${active ? "is-active" : ""}`} aria-current={active ? "page" : undefined}>
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="sidebar__foot">
          <ApiStatus />
        </div>
      </aside>
      <main className="main" id="main">
        {children}
      </main>
    </div>
  );
}
