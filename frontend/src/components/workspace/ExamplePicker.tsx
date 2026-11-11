"use client";

import type { SampleSummary } from "@/lib/api";
import { pct, sci } from "@/lib/format";

/** Bundled held-out test samples (from GET /samples). */
export function ExamplePicker({ samples, selected, onSelect }: { samples: SampleSummary[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="sample-list" role="listbox" aria-label="Benchmark examples" data-testid="example-picker">
      {samples.map((s) => (
        <button
          key={s.id}
          type="button"
          role="option"
          aria-selected={selected === s.id}
          className={`sample-btn ${selected === s.id ? "is-active" : ""}`}
          onClick={() => onSelect(s.id)}
          data-testid={`example-${s.id}`}
          title={`Test sample ${s.test_index}: ${pct(s.fraction_high_permeability, 0)} of cells at a = 1.0; PDE residual of the official target ${sci(s.official_target_pde_residual)}`}
        >
          <div className="sample-btn__id">#{s.test_index}</div>
          <div className="sample-btn__meta">a=1: {pct(s.fraction_high_permeability, 0)}</div>
          <div className="sample-btn__meta">residual {sci(s.official_target_pde_residual)}</div>
        </button>
      ))}
    </div>
  );
}
