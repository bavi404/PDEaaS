#!/usr/bin/env python3
"""Verify that the API inference path (pdeflow.inference.Predictor) IS the evaluated model.

- Loads the committed checkpoint through Predictor with the pinned SHA-256.
- Re-runs all 1,000 test samples for each strategy.
- Recomputes test error against the official targets (primary) and the steady-state reference (secondary).
- Requires the summary statistics to equal reports/m2_eval_fno_baseline.json (relative tolerance 1e-9).
Writes reports/m4_inference_equivalence.json and exits non-zero on any mismatch.

    .venv/bin/python scripts/verify_api_inference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from pdeflow.api.settings import EXPECTED_CHECKPOINT_SHA256
from pdeflow.data.dataset import load_split_arrays
from pdeflow.evaluation.metrics import PRIMARY, SECONDARY, relative_l2, summarize
from pdeflow.inference import Predictor

CASES = {"res64": ("fno_native", 64), "res128": ("fno128_native", 128),
         "res128_interpolation_baseline": ("fno64_interpolated", 128)}


def main() -> int:
    rep = json.loads((ROOT / "reports/m2_eval_fno_baseline.json").read_text())
    p = Predictor(ROOT / "backend/models/fno_darcy_beta1_baseline.pt", EXPECTED_CHECKPOINT_SHA256)
    assert p.sha256 == rep["checkpoint_sha256"], "served checkpoint is not the evaluated checkpoint"
    t = load_split_arrays(ROOT / "data/processed/darcy_beta1.0_test.npz")
    ref = np.load(ROOT / "data/processed/darcy_beta1.0_test_reference.npz")["u_reference"]
    out: dict = {"checkpoint_sha256": p.sha256, "report": "reports/m2_eval_fno_baseline.json", "rtol": 1e-9, "cases": {}}
    ok = True
    for key, (strategy, n) in CASES.items():
        a, off, rf = (t["a"], t["u"], ref) if n == 128 else (t["a"][:, ::2, ::2], t["u"][:, ::2, ::2], ref[:, ::2, ::2])
        pred = p.predict(a, strategy).u
        got = {PRIMARY: summarize(relative_l2(pred, off)), SECONDARY: summarize(relative_l2(pred, rf))}
        case = {"strategy": strategy, "resolution": n, "max_abs_rel_diff": 0.0, "api": got,
                "report": {s: rep[key][s]["relative_l2"] for s in (PRIMARY, SECONDARY)}}
        for s in (PRIMARY, SECONDARY):
            for stat, v in got[s].items():
                r = rep[key][s]["relative_l2"][stat]
                d = abs(v - r) / max(abs(r), 1e-300)
                case["max_abs_rel_diff"] = max(case["max_abs_rel_diff"], d)
                ok &= d <= 1e-9
        out["cases"][key] = case
        print(f"{key:32s} {strategy:20s} primary mean {got[PRIMARY]['mean']:.6f} "
              f"(report {rep[key][PRIMARY]['relative_l2']['mean']:.6f}) max rel diff {case['max_abs_rel_diff']:.2e}")
    out["all_match"] = bool(ok)
    (ROOT / "reports/m4_inference_equivalence.json").write_text(json.dumps(out, indent=2))
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
