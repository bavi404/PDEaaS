#!/usr/bin/env python3
"""Functional + resource check of a running PDEFlow container (docs/DOCKER.md §4).

    python scripts/container_check.py --url http://127.0.0.1:7860 --container pdeflow-m6

Everything goes through the PUBLIC port (Next.js proxy -> internal FastAPI), exactly as a browser would.
Writes reports/m6_container_check.json; exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT / "reports/m2_eval_fno_baseline.json").read_text())
PRIMARY = "primary_vs_official_pdebench_target"


def call(url: str, body: dict | None = None, raw: bytes | None = None) -> tuple[int, dict, float]:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), (time.perf_counter() - t0) * 1000


def cgroup_bytes(container: str, files: list[str]) -> int | None:
    for f in files:
        r = subprocess.run(["docker", "exec", container, "cat", f], check=False, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            return int(r.stdout.strip())
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--container", default="pdeflow-m6", help='docker container name, or "none" for a remote URL')
    ap.add_argument("--out", default=str(ROOT / "reports/m6_container_check.json"))
    a = ap.parse_args()
    api = a.url.rstrip("/") + "/api"
    checks: list[dict] = []

    def check(name: str, ok: bool, **detail):
        checks.append({"check": name, "ok": bool(ok), **detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(detail)[:160]}")

    with urllib.request.urlopen(a.url, timeout=30) as r:
        html = r.read().decode()
        check("frontend loads", r.status == 200 and "PDEFlow" in html, status=r.status)

    s, h, _ = call(f"{api}/health")
    check("health", s == 200 and h.get("status") == "ok", status=s, checkpoint_sha256=h.get("checkpoint_sha256"))
    check("served checkpoint is the evaluated one", h.get("checkpoint_sha256") == REPORT["checkpoint_sha256"])

    s, m, _ = call(f"{api}/model")
    check("model card", s == 200 and m["checkpoint"]["n_params"] == 357217 and m["test_metrics"] is not None,
          n_params=m.get("checkpoint", {}).get("n_params"))

    s, samples, _ = call(f"{api}/samples")
    check("example list", s == 200 and len(samples) == 24, n=len(samples) if s == 200 else None)

    s, r, ms = call(f"{api}/predict", {"sample_id": "test-4701", "resolution": 128, "strategy": "fno128_native"})
    got = r.get("evaluation", {}).get(PRIMARY, {}).get("relative_l2")
    want = REPORT["res128"][PRIMARY]["relative_l2"]["max"]
    check("sample prediction reproduces the evaluated worst case (rel 1e-6)",
          s == 200 and got is not None and abs(got - want) / want < 1e-6, got=got, report=want)

    s, r, ms = call(f"{api}/predict", {"sample_id": "test-2113", "resolution": 64, "return_targets": True})
    check("sample prediction 64x64 with separate targets",
          s == 200 and r["input_kind"] == "benchmark_example" and len(r["prediction"]) == 64
          and set(r["targets"]) == {"official_pdebench_target", "independent_steady_state_reference"},
          rel_l2_official=r.get("evaluation", {}).get(PRIMARY, {}).get("relative_l2"))

    field = [[1.0 if ((i // 16) + (j // 16)) % 2 else 0.1 for j in range(128)] for i in range(128)]
    s, r, ms = call(f"{api}/predict", {"permeability": field})
    check("custom permeability prediction (no targets, no metrics)",
          s == 200 and r["input_kind"] == "custom" and "evaluation" not in r and "targets" not in r
          and r["strategy"] == "fno64_interpolated" and len(r["prediction"]) == 128, warnings=r.get("warnings"))

    s, r, _ = call(f"{api}/predict", {"permeability": [[0.5] * 64] * 64})
    check("out-of-distribution warning", s == 200 and any("non-binary" in w for w in r["warnings"]))

    s, r, _ = call(f"{api}/predict", {"permeability": [[1, 2], [3, 4]]})
    check("malformed input rejected (422 with request id)", s == 422 and bool(r.get("request_id")), detail=r.get("detail"))
    s, r, _ = call(f"{api}/predict", raw=b"{not json")
    check("invalid JSON rejected", s == 422, status=s)
    s, r, _ = call(f"{api}/samples/test-0")
    check("unknown sample 404", s == 404, status=s)

    # latency (sequential, through the public port)
    lat = {}
    for name, body in {"sample_64": {"sample_id": "test-2113", "resolution": 64},
                       "sample_128_default": {"sample_id": "test-2113", "resolution": 128},
                       "custom_128_default": {"permeability": field}}.items():
        wall, server = [], []
        for _ in range(20):
            s, r, ms = call(f"{api}/predict", body)
            wall.append(ms)
            server.append(r["inference_ms"])
        lat[name] = {"n": 20, "wall_ms_median": round(statistics.median(wall), 1),
                     "wall_ms_max": round(max(wall), 1), "server_inference_ms_median": round(statistics.median(server), 1)}
    print("latency", json.dumps(lat))

    # concurrency burst: 32 requests, 8 at a time
    def one(_):
        return call(f"{api}/predict", {"permeability": field})[0]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(one, range(32)))
    check("32 concurrent custom predictions (8 in flight)", codes.count(200) == 32,
          wall_s=round(time.perf_counter() - t0, 2))

    if a.container == "none":  # remote deployment: HTTP checks only
        out = {"url": a.url, "container": None, "checks": checks, "latency": lat, "all_passed": all(c["ok"] for c in checks)}
        Path(a.out).write_text(json.dumps(out, indent=2))
        print("ALL PASSED" if out["all_passed"] else "FAILURES")
        return 0 if out["all_passed"] else 1

    logs = subprocess.run(["docker", "logs", a.container], check=False, capture_output=True, text=True)
    lines = (logs.stdout + logs.stderr).splitlines()
    req_logs = [json.loads(l) for l in lines if l.startswith("{") and '"event": "request"' in l]
    pred_logs = [l for l in lines if '"event": "prediction"' in l]
    check("structured request logs with request ids", len(req_logs) > 50 and all(r.get("request_id") for r in req_logs),
          n_request_logs=len(req_logs), n_prediction_logs=len(pred_logs))

    mem_now = cgroup_bytes(a.container, ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"])
    mem_peak = cgroup_bytes(a.container, ["/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"])
    limit = cgroup_bytes(a.container, ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"])
    tier = 16 * 1024**3
    check("memory peak within HF CPU basic (16 GB)", mem_peak is not None and mem_peak < tier,
          peak_mib=round(mem_peak / 2**20) if mem_peak else None, current_mib=round(mem_now / 2**20) if mem_now else None)

    health = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", a.container], check=False, capture_output=True, text=True).stdout.strip()
    check("docker HEALTHCHECK healthy", health == "healthy", status=health)

    out = {"url": a.url, "container": a.container, "checks": checks, "latency": lat,
           "memory": {"current_bytes": mem_now, "peak_bytes": mem_peak, "cgroup_limit_bytes": limit,
                      "note": "cgroup memory of the whole container (FastAPI + Next.js), peak over its lifetime "
                              "including the e2e suite and the 32-request burst"},
           "all_passed": all(c["ok"] for c in checks)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("ALL PASSED" if out["all_passed"] else "FAILURES")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
