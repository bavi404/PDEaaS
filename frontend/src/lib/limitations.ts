import { PRIMARY, SECONDARY, type ModelCard } from "./api";
import { pct } from "./format";

export const fmt = (v: unknown) => (v === undefined || v === null ? "n/a" : String(v));

/** Limitations derived from the served model card; numbers are read from it, not hard-coded. */
export function limitations(card: ModelCard): string[] {
  const ck = card.checkpoint;
  const tm = card.test_metrics;
  const data = ck.training_config.data ?? {};
  const train = ck.training_config.train ?? {};
  const out = [
    "Trained only on binary permeability fields (a ∈ {0.1, 1.0}) from PDEBench Darcy Flow with forcing β = 1.0, zero Dirichlet boundary conditions and the unit square. Other inputs, forcings or geometries are extrapolation; the API flags out-of-distribution fields.",
    `Trained at ${ck.training_resolution}×${ck.training_resolution} only, on ${fmt(data.n_train)} training samples for ${fmt(train.epochs)} epochs with a single seed (${fmt(ck.training_config.seed)}); no variance across seeds is reported.`,
  ];
  if (tm) {
    const direct = tm.res128_fno128_native[PRIMARY].mean;
    const interp = tm.res128_fno64_interpolated[PRIMARY].mean;
    out.push(
      `Not resolution-invariant within its error level: at 128×128, applying the FNO directly gives ${pct(direct)} mean test error versus ${pct(interp)} for 64×64 inference plus interpolation, so the interpolated path is the default.`,
      `Error depends strongly on the sample: at 64×64 the median is ${pct(tm.res64_fno_native[PRIMARY].median)} but the worst test sample reaches ${pct(tm.res64_fno_native[PRIMARY].max)}; errors are largest for fields dominated by low permeability and near the boundary.`,
      `The official PDEBench targets are not fully converged to steady state for some samples; the model reproduces the official targets (its benchmark), so its error against the exact steady state is higher (${pct(tm.res64_fno_native[SECONDARY].mean)} vs ${pct(tm.res64_fno_native[PRIMARY].mean)} mean at 64×64).`,
    );
  }
  out.push("Inference runs on CPU; latency depends on the host and concurrent load. Results are not comparable to PDEBench's published nRMSE (different split, metric and training budget).");
  return out;
}
