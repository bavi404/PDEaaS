/**
 * Permeability-field utilities for custom inputs.
 * Validation mirrors the backend (backend/pdeflow/api/app.py `validate_permeability`), so most mistakes are caught
 * before a request is sent; the backend remains the authority and its 422 messages are shown verbatim.
 * Grids are indexed [x][y] (first index = x), cell centres at (i + 0.5) / n, matching the API.
 */

import type { Grid, Resolution } from "./api";

export const SUPPORTED_RESOLUTIONS: Resolution[] = [64, 128];
export const TRAINING_VALUES = [0.1, 1.0] as const;
export const MAX_PERMEABILITY = 1e6;

export interface FieldIssue {
  ok: false;
  error: string;
}
export interface FieldOk {
  ok: true;
  grid: Grid;
  resolution: Resolution;
  warnings: string[];
}
export type FieldCheck = FieldOk | FieldIssue;

/** Distribution-shift warnings, same wording as the API. */
export function distributionWarnings(grid: Grid): string[] {
  let min = Infinity;
  let max = -Infinity;
  let nonBinary = false;
  const f32 = new Float32Array(1);
  const allowed = TRAINING_VALUES.map((v) => {
    f32[0] = v;
    return f32[0];
  });
  for (const row of grid)
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
      f32[0] = v;
      if (!allowed.includes(f32[0])) nonBinary = true;
    }
  const w: string[] = [];
  if (min < TRAINING_VALUES[0] || max > TRAINING_VALUES[1])
    w.push(`values outside the training range [${TRAINING_VALUES[0]}, ${TRAINING_VALUES[1]}]: extrapolation, accuracy unknown`);
  if (nonBinary) w.push("non-binary field: training data contain only the values 0.1 and 1.0; accuracy unknown");
  return w;
}

export function validateGrid(value: unknown): FieldCheck {
  if (!Array.isArray(value) || value.length === 0) return { ok: false, error: "Expected a 2-D array of numbers." };
  const n = value.length;
  if (!value.every((r) => Array.isArray(r))) return { ok: false, error: "Every row must be an array of numbers." };
  const lens = new Set((value as unknown[][]).map((r) => r.length));
  if (lens.size !== 1 || !lens.has(n))
    return { ok: false, error: `The field must be square; got ${n} rows with lengths ${[...lens].slice(0, 4).join(", ")}.` };
  if (!SUPPORTED_RESOLUTIONS.includes(n as Resolution))
    return { ok: false, error: `Size ${n}×${n} is not supported. Use 64×64 or 128×128 (the resolutions with measured error).` };
  const grid: Grid = [];
  for (const row of value as unknown[][]) {
    const out: number[] = [];
    for (const v of row) {
      const x = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
      if (!Number.isFinite(x)) return { ok: false, error: "The field contains non-numeric or non-finite values." };
      if (x <= 0) return { ok: false, error: "Permeability must be strictly positive." };
      if (x > MAX_PERMEABILITY) return { ok: false, error: `Values above ${MAX_PERMEABILITY.toExponential(0)} are rejected.` };
      out.push(x);
    }
    grid.push(out);
  }
  return { ok: true, grid, resolution: n as Resolution, warnings: distributionWarnings(grid) };
}

/** Parse pasted/uploaded text: a JSON 2-D array, or CSV/whitespace rows (row i = x index i). */
export function parseField(text: string): FieldCheck {
  const t = text.trim();
  if (!t) return { ok: false, error: "No data provided." };
  if (t.startsWith("[")) {
    try {
      return validateGrid(JSON.parse(t));
    } catch {
      return { ok: false, error: "Invalid JSON: expected a 2-D array such as [[0.1, 1.0], [1.0, 0.1]]." };
    }
  }
  const rows = t
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"))
    .map((l) => l.split(/[\s,;]+/).filter(Boolean));
  return validateGrid(rows);
}

/** Deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function blur1d(src: Float64Array, n: number, sigma: number, horizontal: boolean): Float64Array {
  const r = Math.max(1, Math.ceil(3 * sigma));
  const k = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
  const ks = k.reduce((a, b) => a + b, 0);
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let d = -r; d <= r; d++) {
        // periodic wrap keeps the statistics uniform up to the boundary
        const ii = horizontal ? i : (i + d + n) % n;
        const jj = horizontal ? (j + d + n) % n : j;
        s += src[ii * n + jj] * k[d + r];
      }
      out[i * n + j] = s / ks;
    }
  return out;
}

export interface GeneratorParams {
  resolution: Resolution;
  seed: number;
  /** correlation length as a fraction of the domain (0.02–0.3) */
  lengthScale: number;
  /** fraction of cells set to the high value (0–1) */
  highFraction: number;
  low: number;
  high: number;
}

/**
 * Two-phase random medium: smoothed white noise thresholded at a quantile. Qualitatively similar to the
 * thresholded Gaussian random fields of PDEBench Darcy, but NOT drawn from the PDEBench distribution.
 * Deterministic for a given parameter set.
 */
export function generateField(p: GeneratorParams): Grid {
  const n = p.resolution;
  const rnd = mulberry32(p.seed);
  let z = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    // Box–Muller
    const u1 = Math.max(rnd(), 1e-12);
    const u2 = rnd();
    z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  const sigma = Math.max(0.5, p.lengthScale * n);
  z = blur1d(blur1d(z, n, sigma, true), n, sigma, false);
  const sorted = Float64Array.from(z).sort();
  const frac = Math.min(1, Math.max(0, p.highFraction));
  const q = frac >= 1 ? -Infinity : frac <= 0 ? Infinity : sorted[Math.floor((1 - frac) * (n * n - 1))];
  const grid: Grid = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) row.push(z[i * n + j] > q ? p.high : p.low);
    grid.push(row);
  }
  return grid;
}

export function fieldToCsv(grid: Grid): string {
  return grid.map((r) => r.join(",")).join("\n");
}

export function gridStats(grid: Grid): { min: number; max: number; mean: number } {
  let min = Infinity;
  let max = -Infinity;
  let s = 0;
  let c = 0;
  for (const row of grid)
    for (const v of row) {
      min = Math.min(min, v);
      max = Math.max(max, v);
      s += v;
      c++;
    }
  return { min, max, mean: s / c };
}

export function absDiff(a: Grid, b: Grid): Grid {
  return a.map((row, i) => row.map((v, j) => Math.abs(v - b[i][j])));
}

export function gridRange(...grids: Grid[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const g of grids)
    for (const row of g)
      for (const v of row) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
  return [lo, hi];
}
