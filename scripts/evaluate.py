#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out TEST split (used once, after model selection on val).

Reported separately (never averaged together):
  resolution 64x64  : stride-2 test inputs  -> stride-2 targets
  resolution 128x128: native test inputs    -> native targets (zero-shot; model trained at 64x64)
and for each:
  PRIMARY   vs OFFICIAL PDEBench targets            (key primary_vs_official_pdebench_target)
  SECONDARY vs independent steady-state reference   (key secondary_vs_independent_steady_state_reference)
plus region breakdown (interior / boundary band / last row+col), strata (convergence of the official target,
permeability fraction), a 64->128 linear-interpolation baseline, and CPU inference latency.

    .venv/bin/python scripts/evaluate.py --run runs/fno_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pdeflow.data.dataset import load_split_arrays  # noqa: E402
from pdeflow.data.resample import cell_centres, linear_upsample  # noqa: E402
from pdeflow.evaluation.metrics import (PRIMARY, SECONDARY, evaluate, region_masks, regional_relative_l2,  # noqa: E402
                                        relative_l2, summarize)
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402
from pdeflow.training.core import load_checkpoint, predict, sha256_file  # noqa: E402


def strata(res: np.ndarray, frac: np.ndarray) -> dict[str, np.ndarray]:
    s = {"residual<=1e-3": res <= 1e-3, "1e-3<residual<=1e-2": (res > 1e-3) & (res <= 1e-2),
         "1e-2<residual<=1e-1": (res > 1e-2) & (res <= 1e-1), "residual>1e-1": res > 1e-1}
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)):
        s[f"frac_a1_[{lo},{min(hi, 1.0)}]"] = (frac >= lo) & (frac < hi)
    return s


def regions(pred, off, ref, n, band) -> dict:
    out = {}
    for name, m in region_masks(n, band).items():
        if name == "last_row_col" and n != 128:
            continue
        out[name] = {"n_cells": int(m.sum()), PRIMARY: summarize(regional_relative_l2(pred, off, m)),
                     SECONDARY: summarize(regional_relative_l2(pred, ref, m))}
    return out


def latency(model, res: int, batch: int, reps: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    x = torch.randn(batch, 1, res, res)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return {"resolution": res, "batch": batch, "threads": threads, "reps": reps, "median_ms": float(np.median(ts)),
            "p95_ms": float(np.percentile(ts, 95)), "per_sample_median_ms": float(np.median(ts) / batch)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--test-file", default="data/processed/darcy_beta1.0_test.npz")
    ap.add_argument("--reference-file", default="data/processed/darcy_beta1.0_test_reference.npz")
    ap.add_argument("--val-file", default="data/processed/darcy_beta1.0_val.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    name = run.name
    out = Path(args.out or f"reports/m2_eval_{name}.json")
    log = get_logger(f"pdeflow.eval.{name}", run / "eval.jsonl")
    ck_path = run / args.checkpoint
    model, norms, ck = load_checkpoint(ck_path)
    torch.set_num_threads(4)
    log_event(log, "checkpoint_loaded", path=str(ck_path), epoch=ck["epoch"], sha256=sha256_file(ck_path))

    t = load_split_arrays(args.test_file)
    ref_file = np.load(args.reference_file)
    if not np.array_equal(ref_file["indices"], t["indices"]):
        raise RuntimeError("reference / test index mismatch")
    a, off128, res, idx = t["a"], t["u"], t["pde_residual"].astype(np.float64), t["indices"]
    ref128 = ref_file["u_reference"]
    frac = (a == a.max()).mean(axis=(1, 2))
    S = strata(res, frac)

    def run_model(a_in):
        x = torch.from_numpy(norms.a.encode(a_in).astype(np.float32)).unsqueeze(1)
        return predict(model, x, norms).numpy()[:, 0].astype(np.float64)

    t0 = time.monotonic()
    pred64 = run_model(a[:, ::2, ::2])
    pred128 = run_model(a)
    infer_s = time.monotonic() - t0
    off64, ref64 = off128[:, ::2, ::2], ref128[:, ::2, ::2]
    xs, xf = cell_centres(128, 2), cell_centres(128)
    interp128 = linear_upsample(pred64, xs, xf)

    rep: dict = {
        "run": name, "checkpoint": str(ck_path), "checkpoint_sha256": sha256_file(ck_path), "checkpoint_epoch": ck["epoch"],
        "selection_metric_val_rel_l2_official_64": ck["val_rel_l2_official_64"], "n_test": int(len(idx)),
        "test_file_sha256": sha256_file(args.test_file), "reference_file_sha256": sha256_file(args.reference_file),
        "note": "PRIMARY = official PDEBench targets (benchmark). SECONDARY = independent steady-state reference (analysis only).",
        "res64": {"definition": "stride-2 test inputs -> stride-2 targets (training resolution)",
                  **evaluate(pred64, off64, ref64, S), "regions": regions(pred64, off64, ref64, 64, 1)},
        "res128": {"definition": "native 128x128 test inputs -> native targets (zero-shot; trained at 64x64)",
                   **evaluate(pred128, off128, ref128, S), "regions": regions(pred128, off128, ref128, 128, 2)},
        "res128_interpolation_baseline": {"definition": "FNO 64x64 prediction, coordinate-aware bilinear interpolation to 128x128",
                                          **evaluate(interp128, off128, ref128), "regions": regions(interp128, off128, ref128, 128, 2)},
        "consistency_pred128_subsampled_vs_pred64": summarize(relative_l2(pred128[:, ::2, ::2], pred64)),
        "test_inference_wall_s_64_and_128": round(infer_s, 2),
        "strata_sizes": {k: int(m.sum()) for k, m in S.items()},
    }
    # validation (the selection set) recomputed for transparency
    v = load_split_arrays(args.val_file)
    pv = run_model(v["a"][:, ::2, ::2])
    rep["val_res64_primary"] = summarize(relative_l2(pv, v["u"][:, ::2, ::2]))
    log_event(log, "metrics", res64=rep["res64"][PRIMARY]["relative_l2"], res128=rep["res128"][PRIMARY]["relative_l2"],
              res64_ref=rep["res64"][SECONDARY]["relative_l2"], res128_ref=rep["res128"][SECONDARY]["relative_l2"])

    rep["latency_cpu"] = [latency(model, r, b, 50 if b == 1 else 10, 4) for r in (64, 128) for b in (1, 32)]
    log_event(log, "latency", runs=rep["latency_cpu"])

    # figures --------------------------------------------------------------------------------------------
    hist = json.loads((run / "history.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ep = [h["epoch"] for h in hist]
    ax.semilogy(ep, [h["train_rel_l2"] for h in hist], label="train (official, 64x64)")
    ax.semilogy(ep, [h["val_rel_l2_official_64"] for h in hist], label="validation (official, 64x64)")
    ax.axvline(ck["epoch"], color="k", ls=":", lw=1, label=f"best epoch {ck['epoch']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean relative L2")
    ax.legend()
    ax.set_title(f"{name}: training curves")
    fig.savefig(f"docs/figures/m2_{name}_curves.png", dpi=110)
    plt.close(fig)

    e128 = relative_l2(pred128, off128)
    picks = {"median error": int(np.argsort(e128)[len(e128) // 2]), "worst error": int(np.argmax(e128))}
    fig, axs = plt.subplots(len(picks), 5, figsize=(19, 3.8 * len(picks)), constrained_layout=True)
    for r, (lab, i) in enumerate(picks.items()):
        panels = [(a[i], "a (input)", "cividis"), (off128[i], "official target", "viridis"),
                  (pred128[i], "FNO prediction (128x128)", "viridis"), (np.abs(pred128[i] - off128[i]), "|pred - official|", "magma"),
                  (np.abs(pred128[i] - ref128[i]), "|pred - reference|", "magma")]
        for c, (img, title, cmap) in enumerate(panels):
            im = axs[r, c].imshow(img.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap)
            axs[r, c].set_title(f"{title}" + (f"\n{lab}: test idx {int(idx[i])}\nrelL2 {e128[i]:.3f}, official-target residual {res[i]:.1e}" if c == 0 else ""), fontsize=9)
            fig.colorbar(im, ax=axs[r, c], shrink=0.8)
    fig.savefig(f"docs/figures/m2_{name}_examples.png", dpi=100)
    plt.close(fig)
    rep["figures"] = [f"docs/figures/m2_{name}_curves.png", f"docs/figures/m2_{name}_examples.png"]
    rep["example_picks"] = {k: int(idx[i]) for k, i in picks.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    log_event(log, "report_written", path=str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
