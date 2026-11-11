"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export type Resource<T> =
  | { status: "loading"; data?: undefined; error?: undefined }
  | { status: "success"; data: T; error?: undefined }
  | { status: "error"; data?: undefined; error: ApiError };

/**
 * Load data from the API with explicit loading / success / error states and a reload function.
 * `reload()` (e.g. polling) keeps the previous data visible until the new response arrives; a change of `deps`
 * (a different request) resets to the loading state so stale data is never shown for the new request.
 */
export function useResource<T>(load: () => Promise<T>, deps: unknown[] = []): Resource<T> & { reload: () => void } {
  const [state, setState] = useState<Resource<T>>({ status: "loading" });
  const [nonce, setNonce] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;
  const depsKey = JSON.stringify(deps);
  const lastKey = useRef<string | null>(null);

  useEffect(() => {
    let alive = true;
    const sameRequest = lastKey.current === depsKey;
    lastKey.current = depsKey;
    setState((prev) => (sameRequest && prev.status === "success" ? prev : { status: "loading" }));
    loadRef
      .current()
      .then((data) => alive && setState({ status: "success", data }))
      .catch((e: unknown) => {
        if (!alive) return;
        const err = e instanceof ApiError ? e : new ApiError("server", e instanceof Error ? e.message : String(e));
        setState({ status: "error", error: err });
      });
    return () => {
      alive = false;
    };
  }, [nonce, depsKey]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}
