#!/usr/bin/env python3
"""Inspect the downloaded PDEBench Darcy file and record every statistic in a JSON report.

Outputs
  reports/data_inspection.json      all numbers quoted in docs/DATA.md
  docs/figures/darcy_samples.png    representative (a, u) pairs
  docs/figures/darcy_histograms.png value distributions
  logs/m1_inspect.jsonl             structured event log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pdeflow.data.pdebench import inspect_file, read_samples  # noqa: E402
from pdeflow.data.physics import fv_residual, relative_residual, solve_steady_state  # noqa: E402
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402


def q(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {"min": float(x.min()), "p01": float(np.percentile(x, 1)), "p50": float(np.median(x)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/raw/2D_DarcyFlow_beta1.0_Train.hdf5")
    ap.add_argument("--report", default="reports/data_inspection.json")
    ap.add_argument("--figdir", default="docs/figures")
    ap.add_argument("--log", default="logs/m1_inspect.jsonl")
    args = ap.parse_args()
    log = get_logger("pdeflow.inspect", args.log)

    info = inspect_file(args.file)
    log_event(log, "file_opened", path=str(info.path), n_samples=info.n_samples, resolution=info.resolution, beta=info.beta)
    a, u = read_samples(args.file)  # full file: 2 x 655 MB float32, fits in RAM
    log_event(log, "arrays_loaded", a_shape=a.shape, u_shape=u.shape, a_dtype=str(a.dtype), u_dtype=str(u.dtype))

    R = info.resolution
    dx = 1.0 / R
    rep: dict = {"file": info.path.name, "n_samples": info.n_samples, "resolution": R, "beta_attr": info.beta,
                 "shapes": {"nu": list(a.shape), "tensor(after squeezing channel)": list(u.shape)},
                 "coordinates": {"x_first": float(info.x[0]), "x_last": float(info.x[-1]),
                                 "spacing_unique": np.unique(np.round(np.diff(info.x), 9)).tolist(),
                                 "is_cell_centred": bool(np.allclose(info.x, (np.arange(R) + 0.5) / R, atol=1e-7)),
                                 "x_equals_y": bool(np.array_equal(info.x, info.y))}}

    # --- finiteness and value ranges -------------------------------------------------------------
    rep["non_finite"] = {"nu": int((~np.isfinite(a)).sum()), "u": int((~np.isfinite(u)).sum())}
    vals, counts = np.unique(a, return_counts=True)
    rep["nu_unique_values"] = {"n_unique": int(vals.size),
                               "values": [float(v) for v in vals[:20]], "counts": [int(c) for c in counts[:20]]}
    high = a.max()
    frac_high = (a == high).mean(axis=(1, 2))
    rep["nu_stats"] = q(a)
    rep["nu_fraction_high_per_sample"] = q(frac_high)
    rep["u_stats"] = q(u)
    rep["u_per_sample_max"] = q(u.max(axis=(1, 2)))
    rep["u_negative_fraction"] = float((u < 0).mean())
    border = np.concatenate([u[:, 0, :], u[:, -1, :], u[:, :, 0], u[:, :, -1]], axis=1)
    rep["u_boundary_cells"] = q(border)
    rep["u_interior_max_over_boundary_max"] = float(u.max() / np.abs(border).max())
    log_event(log, "value_ranges", nu=rep["nu_stats"], u=rep["u_stats"], nu_unique=rep["nu_unique_values"]["n_unique"])

    # --- duplicates (leakage risk) ---------------------------------------------------------------
    groups: dict[str, list[int]] = {}
    for i, x in enumerate(a):
        groups.setdefault(hashlib.md5(x.tobytes()).hexdigest(), []).append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    h_au = {hashlib.md5(x.tobytes() + y.tobytes()).hexdigest() for x, y in zip(a, u)}
    rep["duplicates"] = {"unique_nu_fields": len(groups), "unique_(nu,u)_pairs": len(h_au),
                         "duplicate_groups": [{"size": len(g), "indices": g,
                                               "nu_unique_values": np.unique(a[g[0]]).tolist(),
                                               "u_rel_l2_to_group_mean": q(np.linalg.norm((u[g] - u[g].mean(0)).reshape(len(g), -1), axis=1)
                                                                          / np.linalg.norm(u[g].mean(0)))} for g in dup]}
    log_event(log, "duplicate_check", unique_nu=len(groups), unique_pairs=len(h_au), group_sizes=[len(g) for g in dup])

    # --- steady-state / PDE consistency (FV stencil fitted to the released data) -----------------
    rr = np.concatenate([relative_residual(a[i:i + 500], u[i:i + 500], info.beta, dx) for i in range(0, len(a), 500)])
    r0 = fv_residual(a[:200], u[:200], info.beta, dx)
    ring = np.ones(r0.shape[1:], bool)
    ring[1:-1, 1:-1] = False
    n_ref = 100
    u_ss, cg = solve_steady_state(a[:n_ref], info.beta, dx)
    dev = np.linalg.norm((u[:n_ref] - u_ss).reshape(n_ref, -1), axis=1) / np.linalg.norm(u_ss.reshape(n_ref, -1), axis=1)
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    rep["pde_residual"] = {
        "definition": "per-sample RMS of (beta + div(a grad u)) / |beta| = |du/dt| of the generator at t_final",
        "per_sample": q(rr),
        "fraction_above": {str(t): float((rr > t).mean()) for t in (0.001, 0.01, 0.05, 0.1)},
        "first200_rms_interior": float(np.sqrt((r0[:, 1:-1, 1:-1] ** 2).mean())),
        "first200_rms_boundary_ring": float(np.sqrt((r0[:, ring] ** 2).mean())),
        "first200_rms_boundary_ring_if_a_ghost_zero": float(np.sqrt((fv_residual(a[:200], u[:200], info.beta, dx, a_ghost="zero")[:, ring] ** 2).mean())),
        "first200_rms_interior_if_a_ghost_zero": float(np.sqrt((fv_residual(a[:200], u[:200], info.beta, dx, a_ghost="zero")[:, 1:-1, 1:-1] ** 2).mean())),
    }
    rep["steady_state_check"] = {
        "method": "batched CG solve of -div(a grad u)=beta with the same stencil (pdeflow.data.physics)",
        "n_samples": n_ref, "cg": cg,
        "rel_l2_data_vs_exact_steady_state": q(dev),
        "fraction_dev_above_1pct": float((dev > 0.01).mean()), "fraction_dev_above_5pct": float((dev > 0.05).mean()),
        "spearman_residual_vs_deviation": float(np.corrcoef(rank(rr[:n_ref]), rank(dev))[0, 1]),
        "spearman_frac_high_vs_deviation": float(np.corrcoef(rank(frac_high[:n_ref]), rank(dev))[0, 1]),
    }
    log_event(log, "pde_residual", **rep["pde_residual"])
    log_event(log, "steady_state_check", **rep["steady_state_check"])

    # --- figures ---------------------------------------------------------------------------------
    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    picks = [int(np.argmin(frac_high)), int(np.argsort(frac_high)[len(frac_high) // 2]), int(np.argmax(frac_high)), 0]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.2), constrained_layout=True)
    ext = [0, 1, 0, 1]
    for j, i in enumerate(picks):
        im = axes[0, j].imshow(a[i].T, origin="lower", extent=ext, cmap="cividis")
        axes[0, j].set_title(f"a(x)  idx={i}\nfrac(a={high:g})={frac_high[i]:.2f}", fontsize=9)
        fig.colorbar(im, ax=axes[0, j], shrink=0.8)
        im = axes[1, j].imshow(u[i].T, origin="lower", extent=ext, cmap="viridis")
        axes[1, j].set_title(f"u(x)  max={u[i].max():.4f}\nrel. residual={rr[i]:.1e}", fontsize=9)
        fig.colorbar(im, ax=axes[1, j], shrink=0.8)
    fig.suptitle("PDEBench 2D Darcy Flow, beta=1.0 (native 128x128). Index 0 + min / median / max high-permeability fraction")
    fig.savefig(fig_dir / "darcy_samples.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), constrained_layout=True)
    ax[0].bar([f"{v:g}" for v in vals[:20]], counts[:20] / counts.sum())
    ax[0].set_title("a(x): value frequency (all cells)")
    ax[1].hist(frac_high, bins=50)
    ax[1].set_title(f"fraction of cells with a={high:g}")
    ax[2].hist(u.max(axis=(1, 2)), bins=50)
    ax[2].set_title("per-sample max u")
    ax[3].scatter(frac_high[:n_ref], dev, s=8)
    ax[3].set_yscale("log")
    ax[3].set_xlabel("fraction of cells with a=1")
    ax[3].set_title("deviation from exact steady state (first 100)")
    fig.savefig(fig_dir / "darcy_histograms.png", dpi=110)
    plt.close(fig)
    rep["figures"] = {"samples": str(fig_dir / "darcy_samples.png"), "sample_indices": picks,
                      "histograms": str(fig_dir / "darcy_histograms.png")}

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rep, indent=2))
    log_event(log, "report_written", path=args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
