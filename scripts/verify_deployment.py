#!/usr/bin/env python3
"""Verify a PUBLIC PDEFlow deployment against the local, committed code and checkpoint (docs/DEPLOYMENT.md §6).

    .venv/bin/python scripts/verify_deployment.py --url https://bavi404-pdeflow.hf.space

- /api/health must report the pinned checkpoint SHA-256 (the evaluated checkpoint).
- /api/model, /api/samples, /api/activity respond and match the local API.
- For held-out samples and both 128x128 strategies plus a custom field, the public /api/predict result is
  compared element-wise with the SAME request served by the local API in-process (same code, same
  checkpoint). Differences are reported as measured (CPU float32 rounding may differ across hardware).
- Latency: cold first request and warm sequential requests through the public URL.
Writes reports/m7_deployment_check.json; exits non-zero on any failed check. No credentials are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import urllib.error
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
PRIMARY = "primary_vs_official_pdebench_target"
SECONDARY = "secondary_vs_independent_steady_state_reference"
REPORT = json.loads((ROOT / "reports/m2_eval_fno_baseline.json").read_text())


def call(url: str, body: dict | None = None, timeout: float = 120) -> tuple[int, dict, float]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "pdeflow-verify"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), (time.perf_counter() - t0) * 1000
        except ValueError:
            return e.code, {"raw": raw[:200].decode(errors="replace")}, (time.perf_counter() - t0) * 1000
    except (urllib.error.URLError, TimeoutError, ValueError) as e:  # unreachable / timeout / non-JSON 200
        return 0, {"error": str(e)[:200]}, (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports/m7_deployment_check.json"))
    ap.add_argument("--tol", type=float, default=1e-5, help="max |public - local| allowed for predictions")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    api = base + "/api"
    checks: list[dict] = []

    def check(name: str, ok: bool, **d):
        checks.append({"check": name, "ok": bool(ok), **d})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(d)[:180]}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
        from pdeflow.api.app import create_app
        from pdeflow.api.settings import EXPECTED_CHECKPOINT_SHA256, Settings

    started = datetime.now(UTC).isoformat(timespec="seconds")
    s, h, cold_ms = call(f"{api}/health")
    check("public /api/health", s == 200 and h.get("status") == "ok", status=s, first_request_ms=round(cold_ms, 1))
    check("public checkpoint SHA-256 == evaluated checkpoint",
          h.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256 == REPORT["checkpoint_sha256"],
          public=h.get("checkpoint_sha256"))

    try:
        with urllib.request.urlopen(urllib.request.Request(base, headers={"User-Agent": "pdeflow-verify"}), timeout=60) as r:
            html = r.read().decode()
            check("public frontend HTML", r.status == 200 and "PDEFlow" in html, status=r.status)
    except (urllib.error.URLError, TimeoutError) as e:
        check("public frontend HTML", False, error=str(e)[:200])

    with TestClient(create_app(Settings())) as local:
        logging.getLogger("pdeflow.api").setLevel(logging.WARNING)  # keep the local reference run quiet
        s, m, _ = call(f"{api}/model")
        lm = local.get("/model").json()
        def card(c: dict) -> dict:  # num_threads is host configuration (2 on HF CPU basic), not model identity
            return {k: v for k, v in c.items() if k != "num_threads"}
        check("public /api/model == local (excluding host thread count)",
              s == 200 and card(m["checkpoint"]) == card(lm["checkpoint"]) and m["test_metrics"] == lm["test_metrics"],
              status=s, public_num_threads=m.get("checkpoint", {}).get("num_threads"))
        s, smp, _ = call(f"{api}/samples")
        check("public /api/samples == local (24 bundled test samples)", s == 200 and smp == local.get("/samples").json(),
              n=len(smp) if s == 200 else None)
        s, act, _ = call(f"{api}/activity")
        check("public /api/activity", s == 200 and "items" in act, status=s)

        field = [[1.0 if ((i // 16) + (j // 16)) % 2 else 0.1 for j in range(128)] for i in range(128)]
        cases = {
            "test-4701 128 fno128_native (report worst case)": {"sample_id": "test-4701", "resolution": 128, "strategy": "fno128_native"},
            "test-4701 128 default": {"sample_id": "test-4701", "resolution": 128},
            "test-2113 64": {"sample_id": "test-2113", "resolution": 64},
            "test-8142 128 default": {"sample_id": "test-8142", "resolution": 128},
            "custom 128 checkerboard": {"permeability": field},
        }
        comparisons = {}
        for name, body in cases.items():
            s, pub, ms = call(f"{api}/predict", body)
            loc = local.post("/predict", json=body).json()
            if s != 200:
                check(f"predict {name}", False, status=s, body=pub)
                continue
            d = float(np.abs(np.array(pub["prediction"]) - np.array(loc["prediction"])).max())
            rec = {"max_abs_diff_vs_local": d, "public_ms": round(ms, 1), "public_inference_ms": pub["inference_ms"],
                   "strategy": pub["strategy"], "input_kind": pub["input_kind"]}
            if "evaluation" in pub:
                rec["public_rel_l2_official"] = pub["evaluation"][PRIMARY]["relative_l2"]
                rec["local_rel_l2_official"] = loc["evaluation"][PRIMARY]["relative_l2"]
                rec["public_rel_l2_reference"] = pub["evaluation"][SECONDARY]["relative_l2"]
            comparisons[name] = rec
            ok = d <= a.tol and pub["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
            if body.get("strategy") == "fno128_native" and body.get("sample_id") == "test-4701":
                want = REPORT["res128"][PRIMARY]["relative_l2"]["max"]
                ok &= abs(rec["public_rel_l2_official"] - want) / want < 1e-6
                rec["report_value"] = want
            if "permeability" in body:
                ok &= "evaluation" not in pub and "targets" not in pub
            check(f"predict {name}: public == local", ok, **{k: v for k, v in rec.items() if k != "strategy"})

    s, r, _ = call(f"{api}/predict", {"permeability": [[1, 2], [3, 4]]})
    check("public validation error (422 with request id)", s == 422 and bool(r.get("request_id")), status=s)

    lat = {}
    for name, body in {"sample_64": {"sample_id": "test-2113", "resolution": 64},
                       "sample_128_default": {"sample_id": "test-2113", "resolution": 128}}.items():
        wall, server = [], []
        for _ in range(10):
            s, r, ms = call(f"{api}/predict", body)
            if s == 200:
                wall.append(ms)
                server.append(r["inference_ms"])
        lat[name] = {"n": len(wall), "wall_ms_median": round(statistics.median(wall), 1) if wall else None,
                     "wall_ms_max": round(max(wall), 1) if wall else None,
                     "server_inference_ms_median": round(statistics.median(server), 1) if server else None}
    print("latency", json.dumps(lat))

    out = {"url": base, "checked_utc": started, "checks": checks, "prediction_comparisons": comparisons,
           "latency_public": lat, "note": "wall times measured from this sandbox through the HF proxy; they include "
                                          "network round-trip, not only inference",
           "all_passed": all(c["ok"] for c in checks)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("ALL PASSED" if out["all_passed"] else "FAILURES")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
