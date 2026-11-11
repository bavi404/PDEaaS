"use client";

/**
 * Renders one prediction.
 * Benchmark example (bundled test sample): permeability | prediction | OFFICIAL PDEBench target | |pred − official|,
 *   primary metric vs the official target, and a SEPARATE secondary-analysis section for the independent
 *   steady-state reference.
 * Custom input: permeability | prediction only. No ground truth exists, so no target, error map or error metric
 *   is shown.
 */

import { useMemo } from "react";
import { PRIMARY, SECONDARY } from "@/lib/api";
import { absDiff, gridRange } from "@/lib/fields";
import { STRATEGY_LABEL, ms, pct, shortSha } from "@/lib/format";
import type { PredictionRecord } from "@/lib/store";
import { Heatmap } from "./Heatmap";
import { Badge, Card, KeyValue, Notice, Stat } from "./ui";

export function ResultsView({ record, serviceVersion }: { record: PredictionRecord; serviceVersion?: string }) {
  const r = record.response;
  const isBenchmark = record.source === "benchmark_example" && !!r.evaluation && !!r.targets;
  const off = r.targets?.official_pdebench_target;
  const ref = r.targets?.independent_steady_state_reference;

  // prediction and official target share one colour range so they can be compared by eye
  const [uMin, uMax] = useMemo(() => (off ? gridRange(r.prediction, off) : gridRange(r.prediction)), [r.prediction, off]);
  const errOff = useMemo(() => (off ? absDiff(r.prediction, off) : null), [r.prediction, off]);
  const errRef = useMemo(() => (ref ? absDiff(r.prediction, ref) : null), [r.prediction, ref]);
  const [refMin, refMax] = useMemo(() => (ref ? gridRange(r.prediction, ref) : [0, 1]), [r.prediction, ref]);
  const errMax = useMemo(() => (errOff && errRef ? Math.max(gridRange(errOff)[1], gridRange(errRef)[1]) : errOff ? gridRange(errOff)[1] : 1), [errOff, errRef]);

  return (
    <div className="stack" data-testid="results-view" data-kind={record.source}>
      <div className="results-meta">
        {isBenchmark ? (
          <Badge tone="accent">Benchmark example · held-out test sample</Badge>
        ) : (
          <Badge>Custom input · no ground truth</Badge>
        )}
        <span className="mono small">{record.label}</span>
      </div>

      {r.warnings.length > 0 && (
        <Notice tone="warning" title="Input outside the training distribution" testId="ood-warning">
          <ul className="plain-list">
            {r.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
          The model was trained only on binary fields with values 0.1 and 1.0; this prediction is an extrapolation.
        </Notice>
      )}

      <div className="stat-row">
        {isBenchmark && r.evaluation && (
          <Stat
            testId="stat-primary"
            label="Relative L2 vs official PDEBench target"
            value={pct(r.evaluation[PRIMARY].relative_l2)}
            hint="Primary benchmark metric for this sample"
          />
        )}
        <Stat testId="stat-latency" label="Inference latency" value={ms(r.inference_ms)} hint="Server-side, CPU" />
        <Stat label="Resolution" value={`${r.resolution}×${r.resolution}`} hint={`Model run at ${r.model_resolution}×${r.model_resolution}`} />
        <Stat label="Checkpoint" value={<span className="mono">{shortSha(r.checkpoint_sha256)}</span>} hint={serviceVersion ? `service v${serviceVersion}` : "SHA-256 prefix"} />
      </div>

      <Card title={isBenchmark ? "Comparison with the official PDEBench target" : "Prediction"} subtitle={STRATEGY_LABEL[r.strategy]}>
        <div className={`plot-grid ${isBenchmark ? "plot-grid--4" : "plot-grid--2"}`}>
          <Heatmap testId="hm-input" title="Permeability a(x, y)" valueLabel="a" colorscale="permeability" z={record.permeability} />
          <Heatmap testId="hm-prediction" title="Predicted pressure û" valueLabel="û" colorscale="pressure" z={r.prediction} zmin={uMin} zmax={uMax} />
          {isBenchmark && off && errOff && (
            <>
              <Heatmap testId="hm-official" title="Official PDEBench target u" valueLabel="u" colorscale="pressure" z={off} zmin={uMin} zmax={uMax} />
              <Heatmap testId="hm-error" title="Absolute error |û − u|" valueLabel="|û − u|" colorscale="error" z={errOff} zmin={0} zmax={errMax} />
            </>
          )}
        </div>
        {!isBenchmark && (
          <Notice tone="info" testId="no-ground-truth">
            Custom inputs have no ground truth, so no target, error map or error metric is shown. To check accuracy,
            use a benchmark example or compare against an independent solver.
          </Notice>
        )}
      </Card>

      {isBenchmark && r.evaluation && ref && errRef && (
        <Card
          testId="secondary-analysis"
          className="card--secondary"
          title="Secondary analysis: independent steady-state reference"
          subtitle="Exact steady state computed by this project (CG solve). Not used for training, model selection or the benchmark number above."
        >
          <div className="stat-row">
            <Stat testId="stat-secondary" label="Relative L2 vs steady-state reference" value={pct(r.evaluation[SECONDARY].relative_l2)} />
            <Stat testId="stat-gap" label="Official target vs reference" value={pct(r.evaluation.official_target_vs_reference.relative_l2)} hint="How far the official target is from steady state" />
          </div>
          <div className="plot-grid plot-grid--2">
            <Heatmap testId="hm-reference" title="Steady-state reference u_ref" valueLabel="u_ref" colorscale="pressure" z={ref} zmin={refMin} zmax={refMax} />
            <Heatmap testId="hm-error-ref" title="Absolute error |û − u_ref|" valueLabel="|û − u_ref|" colorscale="error" z={errRef} zmin={0} zmax={errMax} />
          </div>
        </Card>
      )}

      <Card title="Run details">
        <KeyValue
          rows={[
            ["Input", isBenchmark ? `Bundled test sample ${record.response.sample_id}` : "User-provided field"],
            ["Strategy", <span key="s" className="mono">{r.strategy}</span>],
            ["Input range", <span key="r" className="mono">{r.input_summary.min.toPrecision(3)} – {r.input_summary.max.toPrecision(3)}</span>],
            ["Checkpoint SHA-256", <span key="c" className="mono break">{r.checkpoint_sha256}</span>],
            ["Requested", new Date(record.createdAt).toISOString().replace("T", " ").slice(0, 19) + " UTC"],
          ]}
        />
      </Card>
    </div>
  );
}
