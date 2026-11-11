import type { Strategy } from "./api";

export const pct = (x: number, digits = 2) => `${(x * 100).toFixed(digits)}%`;
export const ms = (x: number) => (x >= 100 ? `${x.toFixed(0)} ms` : `${x.toFixed(1)} ms`);
export const sci = (x: number, digits = 1) => x.toExponential(digits);
export const shortSha = (s: string) => s.slice(0, 12);
export const int = (x: number) => x.toLocaleString("en-US");

export const STRATEGY_LABEL: Record<Strategy, string> = {
  fno_native: "FNO at 64×64 (training resolution)",
  fno64_interpolated: "FNO at 64×64 + bilinear interpolation to 128×128",
  fno128_native: "FNO applied directly at 128×128 (zero-shot)",
};

export function relativeTime(iso: string, now: Date = new Date()): string {
  const s = Math.max(0, Math.round((now.getTime() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(iso).toISOString().slice(0, 16).replace("T", " ") + " UTC";
}
