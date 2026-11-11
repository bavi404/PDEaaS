"use client";

import { PRIMARY, SECONDARY, type TestMetrics, type TargetPair } from "@/lib/api";
import { pct } from "@/lib/format";

const ROWS: { key: keyof Pick<TestMetrics, "res64_fno_native" | "res128_fno64_interpolated" | "res128_fno128_native">; label: string; note: string }[] = [
  { key: "res64_fno_native", label: "64×64", note: "training resolution" },
  { key: "res128_fno64_interpolated", label: "128×128", note: "64×64 + interpolation (default)" },
  { key: "res128_fno128_native", label: "128×128", note: "direct, zero-shot" },
];

/**
 * Held-out test error from the API's model card (reports/m2_eval_fno_baseline.json, served only if it belongs to
 * the loaded checkpoint). Official-target (PRIMARY) and steady-state-reference (SECONDARY) columns are separate.
 */
export function MetricsTable({ metrics, compact = false }: { metrics: TestMetrics; compact?: boolean }) {
  return (
    <div className="table-wrap">
      <table className="table" data-testid="metrics-table">
        <caption className="sr-only">Test-set relative L2 error, {metrics.n_test} samples</caption>
        <thead>
          <tr>
            <th scope="col">Resolution</th>
            <th scope="col" className="num">vs official target · mean</th>
            <th scope="col" className="num">median</th>
            {!compact && <th scope="col" className="num">p90</th>}
            {!compact && <th scope="col" className="num">max</th>}
            <th scope="col" className="num th-secondary">vs steady-state ref. · mean</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(({ key, label, note }) => {
            const m: TargetPair = metrics[key];
            return (
              <tr key={key} data-testid={`metrics-${key}`}>
                <th scope="row">
                  {label}
                  <span className="muted small block">{note}</span>
                </th>
                <td className="num strong">{pct(m[PRIMARY].mean)}</td>
                <td className="num">{pct(m[PRIMARY].median)}</td>
                {!compact && <td className="num">{pct(m[PRIMARY].p90)}</td>}
                {!compact && <td className="num">{pct(m[PRIMARY].max)}</td>}
                <td className="num td-secondary">{pct(m[SECONDARY].mean)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
