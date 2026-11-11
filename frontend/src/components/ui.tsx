"use client";

import type { ReactNode } from "react";
import type { ApiError } from "@/lib/api";

export function PageHeader({ eyebrow, title, children, actions }: { eyebrow?: string; title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {children && <div className="page-header__lead">{children}</div>}
      </div>
      {actions && <div className="page-header__actions">{actions}</div>}
    </header>
  );
}

export function Card({ title, subtitle, children, className = "", actions, testId }: { title?: string; subtitle?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode; testId?: string }) {
  return (
    <section className={`card ${className}`} data-testid={testId}>
      {(title || actions) && (
        <div className="card__head">
          <div>
            {title && <h2 className="card__title">{title}</h2>}
            {subtitle && <p className="card__subtitle">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, testId }: { label: string; value: ReactNode; hint?: ReactNode; testId?: string }) {
  return (
    <div className="stat" data-testid={testId}>
      <div className="stat__label">{label}</div>
      <div className="stat__value">{value}</div>
      {hint && <div className="stat__hint">{hint}</div>}
    </div>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" | "warn" | "ok" | "danger" }) {
  return <span className={`badge badge--${tone}`}>{children}</span>;
}

export function Notice({ tone, title, children, testId }: { tone: "info" | "warning" | "error"; title?: string; children: ReactNode; testId?: string }) {
  return (
    <div className={`notice notice--${tone}`} role={tone === "error" ? "alert" : "status"} data-testid={testId}>
      {title && <strong className="notice__title">{title}</strong>}
      <div>{children}</div>
    </div>
  );
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state state--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({ error, onRetry, what = "data" }: { error: ApiError; onRetry?: () => void; what?: string }) {
  const title = error.kind === "unavailable" ? "API unavailable" : `Could not load ${what}`;
  return (
    <div className="state state--error" role="alert" data-testid="error-state">
      <strong>{title}</strong>
      <p>{error.message}</p>
      {error.requestId && <p className="mono muted small">request id {error.requestId}</p>}
      {onRetry && (
        <button type="button" className="btn btn--secondary" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="state state--empty" data-testid="empty-state">
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Segmented<T extends string | number>({ label, value, options, onChange, name }: { label: string; value: T; options: { value: T; label: string; disabled?: boolean; hint?: string }[]; onChange: (v: T) => void; name: string }) {
  return (
    <fieldset className="segmented">
      <legend>{label}</legend>
      <div className="segmented__options">
        {options.map((o) => (
          <label key={String(o.value)} className={`segmented__opt ${value === o.value ? "is-active" : ""} ${o.disabled ? "is-disabled" : ""}`} title={o.hint}>
            <input type="radio" name={name} value={String(o.value)} checked={value === o.value} disabled={o.disabled} onChange={() => onChange(o.value)} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function KeyValue({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v], i) => (
        <div key={i} className="kv__row">
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
