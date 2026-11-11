#!/usr/bin/env python3
"""End-to-end HTTP latency of a running PDEFlow API (client-side wall time, sequential requests).

    .venv/bin/python scripts/bench_api.py --url http://127.0.0.1:8000 --reps 30
Writes reports/m4_api_latency.json. Includes JSON (de)serialisation of 64x64 / 128x128 grids.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="reports/m4_api_latency.json")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url, timeout=60)
    field128 = c.get("/samples/test-2113", params={"resolution": 128}).json()["permeability"]
    cases = {
        "sample_64_fno_native": {"sample_id": "test-2113", "resolution": 64},
        "sample_128_fno64_interpolated": {"sample_id": "test-2113", "resolution": 128},
        "sample_128_fno128_native": {"sample_id": "test-2113", "resolution": 128, "strategy": "fno128_native"},
        "user_field_128_default": {"permeability": field128},
    }
    out = {"url": args.url, "reps": args.reps, "client": "httpx, sequential, same host",
           "platform": platform.platform(), "cases": {}}
    for name, body in cases.items():
        for _ in range(3):
            c.post("/predict", json=body).raise_for_status()
        wall, server = [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            r = c.post("/predict", json=body)
            wall.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            server.append(r.json()["inference_ms"])
        out["cases"][name] = {"request_bytes": len(json.dumps(body)), "response_bytes": len(r.content),
                              "wall_ms_median": float(np.median(wall)), "wall_ms_p95": float(np.percentile(wall, 95)),
                              "model_inference_ms_median": float(np.median(server))}
        print(name, {k: round(v, 1) if isinstance(v, float) else v for k, v in out["cases"][name].items()})
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
