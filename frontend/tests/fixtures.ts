// Test fixtures shaped like real API responses (used only by unit tests; the app never uses fixtures).
import type { Activity, Grid, ModelCard, PredictResponse, SampleDetail, SampleSummary } from "@/lib/api";

export const grid = (n: number, f: (i: number, j: number) => number): Grid =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => f(i, j)));

const summary = (mean: number) => ({ n: 1000, mean, median: mean * 0.6, p90: mean * 2, p99: mean * 5, max: mean * 8 });

export const MODEL_CARD: ModelCard = {
  service_version: "0.1.0",
  checkpoint: {
    format: "pdeflow-fno-v1",
    sha256: "d7133c7c681cf3e9632579833fe357636d84ff7a5678d94b2f24c38955124df7",
    file: "fno_darcy_beta1_baseline.pt",
    architecture: "neuraloperator FNO (2D)",
    model_config: { n_modes: [12, 12], hidden_channels: 32, n_layers: 4, in_channels: 1, out_channels: 1 },
    n_params: 357217,
    epoch: 139,
    selection_metric: { name: "val_rel_l2_official_64", value: 0.0303 },
    training_resolution: 64,
    supported_resolutions: [64, 128],
    default_strategy: { "64": "fno_native", "128": "fno64_interpolated" },
    normalizers: { a: { mean: 0.54, std: 0.45, eps: 1e-8 }, u: { mean: 0.16, std: 0.14, eps: 1e-8 }, fitted_on: "train" },
    training_config: { seed: 0, train: { epochs: 150, batch_size: 16, lr: 0.001, weight_decay: 0.0001, scheduler: "cosine", loss: "relative_l2" }, data: { n_train: 2000, n_val: 1000, stride: 2 } },
    provenance: { torch: "2.15.0+cpu", python: "3.11.15" },
    num_threads: 4,
  },
  dataset: "PDEBench 2D Darcy Flow beta=1.0, CC BY 4.0. Subset of this project's held-out test split.",
  targets: { official_pdebench_target: "PRIMARY benchmark", independent_steady_state_reference: "SECONDARY analysis only" },
  test_metrics: {
    source: "m2_eval_fno_baseline.json",
    n_test: 1000,
    metric: "per-sample relative L2",
    res64_fno_native: { primary_vs_official_pdebench_target: summary(0.0293), secondary_vs_independent_steady_state_reference: summary(0.0391) },
    res128_fno128_native: { primary_vs_official_pdebench_target: summary(0.0421), secondary_vs_independent_steady_state_reference: summary(0.0512) },
    res128_fno64_interpolated: { primary_vs_official_pdebench_target: summary(0.032), secondary_vs_independent_steady_state_reference: summary(0.0417) },
    note: "",
  },
};

export const SAMPLES: SampleSummary[] = [
  { id: "test-2113", test_index: 2113, fraction_high_permeability: 0.68, official_target_pde_residual: 4.7e-4 },
  { id: "test-4701", test_index: 4701, fraction_high_permeability: 0.002, official_target_pde_residual: 0.45 },
];

export const sampleDetail = (id: string, n: 64 | 128): SampleDetail => ({
  ...SAMPLES.find((s) => s.id === id)!,
  resolution: n,
  permeability: grid(n, (i, j) => ((i + j) % 3 === 0 ? 1 : 0.1)),
  targets: { official_pdebench_target: grid(n, () => 0.2), independent_steady_state_reference: grid(n, () => 0.21) },
});

export const benchmarkResponse = (n: 64 | 128 = 64): PredictResponse => ({
  prediction: grid(n, (i, j) => 0.19 + 0.0001 * i * j),
  resolution: n,
  model_resolution: 64,
  strategy: n === 64 ? "fno_native" : "fno64_interpolated",
  inference_ms: 33.2,
  checkpoint_sha256: MODEL_CARD.checkpoint.sha256,
  warnings: [],
  input_summary: { min: 0.1, max: 1, mean: 0.4, fraction_at_max: 0.33 },
  input_kind: "benchmark_example",
  sample_id: "test-2113",
  evaluation: {
    primary_vs_official_pdebench_target: { relative_l2: 0.0134 },
    secondary_vs_independent_steady_state_reference: { relative_l2: 0.0131 },
    official_target_vs_reference: { relative_l2: 0.0006 },
  },
  targets: { official_pdebench_target: grid(n, () => 0.2), independent_steady_state_reference: grid(n, () => 0.21) },
});

export const customResponse = (n: 64 | 128 = 64, warnings: string[] = []): PredictResponse => ({
  prediction: grid(n, () => 0.1),
  resolution: n,
  model_resolution: 64,
  strategy: n === 64 ? "fno_native" : "fno64_interpolated",
  inference_ms: 27.1,
  checkpoint_sha256: MODEL_CARD.checkpoint.sha256,
  warnings,
  input_summary: { min: 0.1, max: 1, mean: 0.5, fraction_at_max: 0.5 },
  input_kind: "custom",
});

export const ACTIVITY: Activity = {
  items: [
    { timestamp_utc: "2026-11-11T12:00:00+00:00", input_kind: "benchmark_example", sample_id: "test-2113", resolution: 64, strategy: "fno_native", inference_ms: 30, n_warnings: 0, rel_l2_vs_official: 0.0134 },
    { timestamp_utc: "2026-11-11T11:59:00+00:00", input_kind: "custom", sample_id: null, resolution: 128, strategy: "fno64_interpolated", inference_ms: 40, n_warnings: 1, rel_l2_vs_official: null },
  ],
  total_since_start: 2,
  started_utc: "2026-11-11T11:00:00+00:00",
  note: "",
};
