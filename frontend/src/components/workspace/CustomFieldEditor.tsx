"use client";

import { useState } from "react";
import type { Resolution } from "@/lib/api";
import { parseField, type FieldCheck, type GeneratorParams } from "@/lib/fields";
import { Segmented } from "../ui";

export type CustomSource = "generator" | "paste";

export interface CustomFieldEditorProps {
  source: CustomSource;
  onSourceChange: (s: CustomSource) => void;
  params: GeneratorParams;
  onParamsChange: (p: GeneratorParams) => void;
  pasted: FieldCheck | null;
  onPasted: (c: FieldCheck | null) => void;
}

function NumberField({ label, value, step, min, max, onChange, testId }: { label: string; value: number; step: number; min: number; max?: number; onChange: (v: number) => void; testId?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        className="input"
        type="number"
        value={Number.isFinite(value) ? value : ""}
        step={step}
        min={min}
        max={max}
        data-testid={testId}
        onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
      />
    </label>
  );
}

/** Custom permeability input: seeded two-phase generator, or paste / upload of a JSON or CSV grid. */
export function CustomFieldEditor({ source, onSourceChange, params, onParamsChange, pasted, onPasted }: CustomFieldEditorProps) {
  const [text, setText] = useState("");
  const set = (patch: Partial<GeneratorParams>) => onParamsChange({ ...params, ...patch });

  const ingest = (t: string) => {
    setText(t);
    onPasted(t.trim() ? parseField(t) : null);
  };

  return (
    <div className="stack" style={{ gap: 14 }}>
      <Segmented<CustomSource>
        label="Field source"
        name="custom-source"
        value={source}
        onChange={onSourceChange}
        options={[
          { value: "generator", label: "Random two-phase medium" },
          { value: "paste", label: "Paste or upload" },
        ]}
      />
      {source === "generator" ? (
        <div className="stack" style={{ gap: 12 }} data-testid="generator">
          <p className="small muted">
            Smoothed noise thresholded into two phases. Resembles the benchmark inputs but is not drawn from the
            PDEBench distribution.
          </p>
          <div className="form-grid">
            <NumberField label="Seed" value={params.seed} step={1} min={0} onChange={(v) => set({ seed: Math.max(0, Math.floor(v || 0)) })} testId="gen-seed" />
            <label className="field">
              <span>Correlation length · {params.lengthScale.toFixed(2)}</span>
              <input type="range" min={0.02} max={0.3} step={0.01} value={params.lengthScale} onChange={(e) => set({ lengthScale: Number(e.target.value) })} aria-label="Correlation length" />
            </label>
            <label className="field">
              <span>High-permeability fraction · {Math.round(params.highFraction * 100)}%</span>
              <input type="range" min={0} max={1} step={0.05} value={params.highFraction} onChange={(e) => set({ highFraction: Number(e.target.value) })} aria-label="High-permeability fraction" />
            </label>
            <div />
            <NumberField label="Low value" value={params.low} step={0.05} min={0.001} onChange={(v) => set({ low: v })} testId="gen-low" />
            <NumberField label="High value" value={params.high} step={0.05} min={0.001} onChange={(v) => set({ high: v })} testId="gen-high" />
          </div>
          <p className="small muted">Training data use low = 0.1 and high = 1.0. Other values are allowed but flagged as out of distribution.</p>
        </div>
      ) : (
        <div className="stack" style={{ gap: 10 }} data-testid="paste">
          <label className="field">
            <span>64×64 or 128×128 grid: JSON array, or CSV with one row per x index</span>
            <textarea className="textarea" value={text} onChange={(e) => ingest(e.target.value)} placeholder="[[0.1, 1.0, …], …]  or  0.1,1.0,…" data-testid="paste-input" spellCheck={false} />
          </label>
          <label className="btn btn--secondary" style={{ alignSelf: "flex-start" }}>
            Upload .json / .csv / .txt
            <input
              type="file"
              accept=".json,.csv,.txt,application/json,text/csv,text/plain"
              hidden
              data-testid="file-input"
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (f) ingest(await f.text());
                e.target.value = "";
              }}
            />
          </label>
          {pasted && !pasted.ok && (
            <p className="notice notice--error" role="alert" data-testid="paste-error">
              {pasted.error}
            </p>
          )}
          {pasted && pasted.ok && (
            <p className="small muted" data-testid="paste-ok">
              Parsed a {pasted.resolution}×{pasted.resolution} field.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export const DEFAULT_GENERATOR: Omit<GeneratorParams, "resolution"> & { resolution: Resolution } = {
  resolution: 128,
  seed: 7,
  lengthScale: 0.08,
  highFraction: 0.5,
  low: 0.1,
  high: 1.0,
};
