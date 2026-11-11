#!/usr/bin/env python3
"""Bundle a small, fixed subset of TEST samples for the API's /samples endpoints.

Each sample keeps its two targets in separately named arrays:
  u_official   official PDEBench `tensor` (unmodified)         -> PRIMARY benchmark target
  u_reference  independent steady-state CG reference (ours)    -> SECONDARY analysis only
Only test samples are bundled: they were never used for training, normalisation or model selection.
Selection is deterministic:
  - the two samples shown in docs/figures/m2_fno_baseline_examples.png (test idx 2113 median, 4701 worst)
  - 22 others chosen with numpy default_rng(0)

PDEBench data: Takamoto et al., DaRUS doi:10.18419/darus-2986, licensed CC BY 4.0.

    .venv/bin/python scripts/build_demo_samples.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend/assets/demo_samples_v1.npz"
N_RANDOM = 22
FIXED = (2113, 4701)
ATTRIBUTION = ("PDEBench 2D Darcy Flow beta=1.0 (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks), "
               "DaRUS doi:10.18419/darus-2986, CC BY 4.0. Subset of this project's held-out test split; "
               "u_reference computed by this project (docs/DATA.md §5).")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    tf, rf = ROOT / "data/processed/darcy_beta1.0_test.npz", ROOT / "data/processed/darcy_beta1.0_test_reference.npz"
    t, r = np.load(tf), np.load(rf)
    idx = t["indices"]
    if not np.array_equal(idx, r["indices"]):
        raise RuntimeError("test/reference index mismatch")
    pos_fixed = [int(np.flatnonzero(idx == i)[0]) for i in FIXED]
    rest = np.setdiff1d(np.arange(len(idx)), pos_fixed)
    pos = sorted(pos_fixed + list(np.random.default_rng(0).choice(rest, N_RANDOM, replace=False)))
    a = t["a"][pos]
    if not np.isin(a, np.float32([0.1, 1.0])).all():
        raise RuntimeError("unexpected permeability values")
    meta = {"attribution": ATTRIBUTION, "license": "CC BY 4.0", "split": "test",
            "test_file_sha256": sha(tf), "reference_file_sha256": sha(rf),
            "selection": f"fixed {list(FIXED)} + default_rng(0) choice of {N_RANDOM}",
            "u_official": "official PDEBench target (PRIMARY)",
            "u_reference": "independent steady-state reference (SECONDARY, analysis only)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, indices=idx[pos], a=a, u_official=t["u"][pos], u_reference=r["u_reference"][pos],
                        official_target_residual=t["pde_residual"][pos], meta=np.array(json.dumps(meta)))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), test indices {idx[pos].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
