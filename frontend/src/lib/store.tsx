"use client";

/**
 * Holds the most recent prediction so the Results page can render it after navigation.
 * Persisted in sessionStorage (per tab) as a convenience only; every value shown comes from the API response.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Grid, PredictRequest, PredictResponse } from "./api";

export interface PredictionRecord {
  createdAt: string;
  /** "benchmark_example": bundled test sample with official target; "custom": user field, no ground truth. */
  source: "benchmark_example" | "custom";
  label: string;
  permeability: Grid;
  request: PredictRequest;
  response: PredictResponse;
}

interface Store {
  latest: PredictionRecord | null;
  setLatest: (r: PredictionRecord | null) => void;
  hydrated: boolean;
}

const KEY = "pdeflow.latestPrediction.v1";
const Ctx = createContext<Store | null>(null);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [latest, setLatestState] = useState<PredictionRecord | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = window.sessionStorage.getItem(KEY);
      if (raw) setLatestState(JSON.parse(raw) as PredictionRecord);
    } catch {
      /* storage unavailable or corrupt: start empty */
    }
    setHydrated(true);
  }, []);

  const setLatest = useCallback((r: PredictionRecord | null) => {
    setLatestState(r);
    try {
      if (r) window.sessionStorage.setItem(KEY, JSON.stringify(r));
      else window.sessionStorage.removeItem(KEY);
    } catch {
      /* quota or privacy mode: keep in memory only */
    }
  }, []);

  const value = useMemo(() => ({ latest, setLatest, hydrated }), [latest, setLatest, hydrated]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function usePredictionStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("usePredictionStore must be used inside PredictionProvider");
  return s;
}
