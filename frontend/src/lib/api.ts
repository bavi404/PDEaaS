/**
 * Typed client for the PDEFlow FastAPI backend (docs/API.md).
 * The browser calls same-origin `/api/*`; the Next.js route handler (src/app/api/[...path]/route.ts) forwards to FastAPI.
 * Types mirror backend/pdeflow/api/schemas.py. The two targets keep their distinct names everywhere:
 *   official_pdebench_target            -> PRIMARY benchmark target (model trained/selected on it)
 *   independent_steady_state_reference  -> SECONDARY analysis only
 */

export type Grid = number[][];
export type Strategy = "fno_native" | "fno64_interpolated" | "fno128_native";
export type Resolution = 64 | 128;
export type InputKind = "benchmark_example" | "custom";

export const PRIMARY = "primary_vs_official_pdebench_target" as const;
export const SECONDARY = "secondary_vs_independent_steady_state_reference" as const;

export interface Health {
  status: "ok";
  model_loaded: boolean;
  checkpoint_sha256: string;
  n_samples: number;
}

export interface ErrorSummary {
  n: number;
  mean: number;
  median: number;
  p90: number;
  p99: number;
  max: number;
}

export interface TargetPair {
  [PRIMARY]: ErrorSummary;
  [SECONDARY]: ErrorSummary;
}

export interface TestMetrics {
  source: string;
  n_test: number;
  metric: string;
  res64_fno_native: TargetPair;
  res128_fno128_native: TargetPair;
  res128_fno64_interpolated: TargetPair;
  note: string;
}

export interface ScalarNorm {
  mean: number;
  std: number;
  eps: number;
}

export interface CheckpointInfo {
  format: string;
  sha256: string;
  file: string;
  architecture: string;
  model_config: { n_modes: number[]; hidden_channels: number; n_layers: number; in_channels: number; out_channels: number };
  n_params: number;
  epoch: number;
  selection_metric: { name: string; value: number };
  training_resolution: number;
  supported_resolutions: number[];
  default_strategy: Record<string, Strategy>;
  normalizers: { a: ScalarNorm; u: ScalarNorm; fitted_on: string };
  training_config: {
    seed?: number;
    train?: Record<string, unknown> & { epochs?: number; batch_size?: number; lr?: number; weight_decay?: number; scheduler?: string; loss?: string };
    data?: Record<string, unknown> & { n_train?: number; n_val?: number; stride?: number };
  };
  provenance: Record<string, unknown>;
  num_threads: number;
}

export interface ModelCard {
  service_version: string;
  checkpoint: CheckpointInfo;
  dataset: string;
  targets: { official_pdebench_target: string; independent_steady_state_reference: string };
  test_metrics: TestMetrics | null;
}

export interface SampleSummary {
  id: string;
  test_index: number;
  fraction_high_permeability: number;
  official_target_pde_residual: number;
}

export interface Targets {
  official_pdebench_target: Grid;
  independent_steady_state_reference: Grid;
}

export interface SampleDetail extends SampleSummary {
  resolution: Resolution;
  permeability: Grid;
  targets: Targets;
}

export interface SampleEvaluation {
  [PRIMARY]: { relative_l2: number };
  [SECONDARY]: { relative_l2: number };
  official_target_vs_reference: { relative_l2: number };
}

export interface PredictRequest {
  permeability?: Grid;
  sample_id?: string;
  resolution?: Resolution;
  strategy?: Strategy | null;
  return_targets?: boolean;
}

export interface PredictResponse {
  prediction: Grid;
  resolution: Resolution;
  model_resolution: number;
  strategy: Strategy;
  inference_ms: number;
  checkpoint_sha256: string;
  warnings: string[];
  input_summary: { min: number; max: number; mean: number; fraction_at_max: number };
  input_kind: InputKind;
  sample_id?: string;
  evaluation?: SampleEvaluation;
  targets?: Targets;
}

export interface ActivityItem {
  timestamp_utc: string;
  input_kind: InputKind;
  sample_id: string | null;
  resolution: number;
  strategy: Strategy;
  inference_ms: number;
  n_warnings: number;
  rel_l2_vs_official: number | null;
}

export interface Activity {
  items: ActivityItem[];
  total_since_start: number;
  started_utc: string;
  note: string;
}

export type ApiErrorKind = "unavailable" | "validation" | "not_found" | "too_large" | "server" | "malformed";

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    message: string,
    public status?: number,
    public requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Turn FastAPI's `detail` (string or list of pydantic errors) into one readable message. */
export function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const loc = Array.isArray(d?.loc) ? d.loc.filter((x: unknown) => x !== "body").join(".") : "";
        return loc ? `${loc}: ${d?.msg ?? "invalid"}` : String(d?.msg ?? "invalid");
      })
      .join("; ");
  }
  return "Unexpected error response";
}

function kindFor(status: number): ApiErrorKind {
  if (status === 422 || status === 400) return "validation";
  if (status === 404) return "not_found";
  if (status === 413) return "too_large";
  if (status === 502 || status === 503 || status === 504) return "unavailable";
  return "server";
}

export const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit, fetcher: typeof fetch = fetch): Promise<T> {
  let res: Response;
  try {
    res = await fetcher(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("unavailable", "The PDEFlow API could not be reached. Check that the backend is running.");
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    if (!res.ok) throw new ApiError(kindFor(res.status), `API request failed (HTTP ${res.status})`, res.status);
    throw new ApiError("malformed", "The API returned a response that is not valid JSON.", res.status);
  }
  if (!res.ok) {
    const b = body as { detail?: unknown; request_id?: string };
    throw new ApiError(kindFor(res.status), formatDetail(b?.detail), res.status, b?.request_id);
  }
  return body as T;
}

export function createApi(fetcher: typeof fetch = (...a) => fetch(...a)) {
  return {
    health: () => request<Health>("/health", undefined, fetcher),
    model: () => request<ModelCard>("/model", undefined, fetcher),
    samples: () => request<SampleSummary[]>("/samples", undefined, fetcher),
    sample: (id: string, resolution: Resolution) =>
      request<SampleDetail>(`/samples/${encodeURIComponent(id)}?resolution=${resolution}`, undefined, fetcher),
    activity: (limit = 10) => request<Activity>(`/activity?limit=${limit}`, undefined, fetcher),
    predict: (body: PredictRequest) =>
      request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify(body) }, fetcher),
  };
}

export const api = createApi();
export type Api = ReturnType<typeof createApi>;
