#!/usr/bin/env python3
"""Scientific audit of the PDEBench Darcy beta=1.0 targets: duplicate inputs and steady-state convergence.

Terminology used throughout:
  * OFFICIAL target  - `tensor` as released by PDEBench (never modified by this project)
  * REFERENCE        - an independent exact steady state of the discrete PDE, -div(a grad u) = beta,
                       computed here by CG with the stencil fitted to the released data (pdeflow.data.physics)
  * deviation        - rel-L2(official, reference) = ||u_off - u_ref|| / ||u_ref||

Outputs
  reports/data_audit.json
  data/processed/darcy_beta1.0_test_reference.npz   (reference steady states for the test split; not committed)
  docs/figures/convergence_audit.png
  logs/m1_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pdeflow.data.dataset import load_split_arrays  # noqa: E402
from pdeflow.data.pdebench import read_samples  # noqa: E402
from pdeflow.data.physics import relative_residual, solve_steady_state  # noqa: E402
from pdeflow.data.splits import SplitManifest  # noqa: E402
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402


def q(x) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {"n": int(x.size), "min": float(x.min()), "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)), "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()), "mean": float(x.mean())}


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    return np.linalg.norm((x - ref).reshape(len(x), -1), axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)


def rank(x):
    return np.argsort(np.argsort(x))


def spearman(x, y) -> float:
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/raw/2D_DarcyFlow_beta1.0_Train.hdf5")
    ap.add_argument("--splits", default="data/splits/darcy_beta1.0_seed20261111.json")
    ap.add_argument("--test-file", default="data/processed/darcy_beta1.0_test.npz")
    ap.add_argument("--reference-out", default="data/processed/darcy_beta1.0_test_reference.npz")
    ap.add_argument("--report", default="reports/data_audit.json")
    ap.add_argument("--fig", default="docs/figures/convergence_audit.png")
    ap.add_argument("--log", default="logs/m1_audit.jsonl")
    ap.add_argument("--batch", type=int, default=125)
    args = ap.parse_args()
    log = get_logger("pdeflow.audit", args.log)
    m = SplitManifest.load(args.splits)
    R, beta = 128, 1.0
    dx = 1.0 / R
    rep: dict = {"definitions": {
        "official_target": "PDEBench `tensor` as released; unchanged",
        "reference": "exact discrete steady state of -div(a grad u)=beta (CG, rel. residual <= 1e-10), computed by this project",
        "deviation": "||u_official - u_reference||_2 / ||u_reference||_2 per sample",
        "pde_residual": "RMS(beta + div(a grad u_official)) / beta = |du/dt| of the generator at t=2"}}

    # ---- 1. duplicate inputs ------------------------------------------------------------------------
    dups = []
    for g in m.duplicate_groups:
        a, u = read_samples(args.file, g)
        assert all(np.array_equal(a[0], x) for x in a), "group members must be byte-identical"
        u_ref, cg = solve_steady_state(a[0], beta, dx)
        dev = rel(u, np.broadcast_to(u_ref, u.shape))
        pair = np.array([np.linalg.norm(u[i].astype(np.float64) - u[j]) / np.linalg.norm(u_ref) for i, j in combinations(range(len(g)), 2)])
        mean_u = u.astype(np.float64).mean(0)
        to_mean = rel(u, np.broadcast_to(mean_u, u.shape))
        res = relative_residual(a, u, beta, dx)
        # continuum estimate of the slowest decay for constant a on a square of side L = 1 + dx (effective domain)
        a0, L = float(a[0, 0, 0]), 1.0 + dx
        lam1 = a0 * 2 * np.pi**2 / L**2
        home = next(n for n in ("train", "val", "test") if set(g) & set(getattr(m, n)))
        pos = sorted(getattr(m, home).index(i) for i in g)
        dups.append({
            "size": len(g), "indices": g, "field": f"constant a = {a0:g}", "split": home,
            "positions_in_shuffled_split": {"first": pos[0], "last": pos[-1]},
            "included_in_processed_files": False if pos[0] >= 2000 else True,
            "reference_cg": cg,
            "deviation_official_vs_reference": q(dev),
            "pairwise_rel_l2_between_official_targets_(norm_by_reference)": q(pair),
            "rel_l2_to_group_mean": q(to_mean),
            "group_mean_deviation_vs_reference": float(rel(mean_u[None], u_ref[None])[0]),
            "pde_residual": q(res),
            "spearman_residual_vs_deviation": spearman(res, dev),
            "u_max": q(u.max(axis=(1, 2))), "reference_u_max": float(u_ref.max()),
            "official_minus_reference_sign": {"frac_members_with_mean_above_reference": float((u.mean(axis=(1, 2)) > u_ref.mean()).mean())},
            "slowest_mode": {"lambda1_continuum": lam1, "t_final": 2.0, "exp(-lambda1*t_final)": float(np.exp(-lam1 * 2.0))},
        })
        log_event(log, "duplicate_group_audited", size=len(g), deviation_median=float(np.median(dev)),
                  pairwise_median=float(np.median(pair)), residual_dev_spearman=dups[-1]["spearman_residual_vs_deviation"])
    rep["duplicate_groups"] = dups

    # ---- 2. convergence: reference for the full test split ------------------------------------------
    t = load_split_arrays(args.test_file)
    a_t, u_t, idx_t = t["a"], t["u"], t["indices"]
    t0 = time.monotonic()
    refs, infos = [], []
    for s in range(0, len(a_t), args.batch):
        r_, inf = solve_steady_state(a_t[s:s + args.batch], beta, dx)
        refs.append(r_.astype(np.float32))
        infos.append(inf)
        log_event(log, "reference_batch", start=s, n=len(r_), **inf, elapsed_s=round(time.monotonic() - t0, 1))
    u_ref_t = np.concatenate(refs)
    if not all(i["converged"] for i in infos):
        log_event(log, "cg_not_converged", 40)
        return 1
    np.savez(args.reference_out, u_reference=u_ref_t, indices=idx_t, note=np.array(
        "independent CG steady state; NOT the official PDEBench target"),
             target_source=np.array("independent steady-state reference (secondary analysis only; never used for training)"))
    dev_t = rel(u_t, u_ref_t)
    res_t = t["pde_residual"].astype(np.float64)
    frac_t = (a_t == a_t.max()).mean(axis=(1, 2))
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    by_frac = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (frac_t >= lo) & (frac_t < hi if hi < 1 else frac_t <= hi)
        if sel.any():
            by_frac.append({"frac_a1_range": [lo, hi], "n": int(sel.sum()), "deviation_median": float(np.median(dev_t[sel])),
                            "deviation_mean": float(dev_t[sel].mean())})
    # residual -> deviation calibration (log-log least squares) on the test split, applied to all 10k residuals
    A = np.vstack([np.log(res_t), np.ones_like(res_t)]).T
    slope, icpt = np.linalg.lstsq(A, np.log(dev_t), rcond=None)[0]
    pred = np.exp(icpt) * res_t**slope
    a_all, u_all = read_samples(args.file)
    res_all = np.concatenate([relative_residual(a_all[i:i + 500], u_all[i:i + 500], beta, dx) for i in range(0, 10000, 500)])
    est_all = np.exp(icpt) * res_all**slope
    thresholds = (0.001, 0.005, 0.01, 0.05, 0.1)
    rep["convergence_test_split"] = {
        "n": int(len(dev_t)), "cg_batches": infos, "solve_time_s": round(time.monotonic() - t0, 1),
        "deviation_official_vs_reference": q(dev_t),
        "fraction_deviation_above": {str(x): float((dev_t > x).mean()) for x in thresholds},
        "spearman_residual_vs_deviation": spearman(res_t, dev_t),
        "spearman_frac_a1_vs_deviation": spearman(frac_t, dev_t),
        "deviation_by_fraction_a1": by_frac,
        "mean_official_over_mean_reference": float(u_t.mean() / u_ref_t.mean()),
        "residual_to_deviation_fit": {"model": "dev = exp(b) * residual^m", "m": float(slope), "b": float(icpt),
                                      "median_abs_log_error": float(np.median(np.abs(np.log(pred) - np.log(dev_t))))},
    }
    rep["convergence_all_10000_estimated"] = {
        "method": "deviation estimated from each sample's PDE residual with the fit above (an ESTIMATE, not exact solves)",
        "estimated_deviation": q(est_all),
        "estimated_fraction_deviation_above": {str(x): float((est_all > x).mean()) for x in thresholds},
    }
    log_event(log, "convergence_audited", **{k: rep["convergence_test_split"][k] for k in
                                               ("deviation_official_vs_reference", "fraction_deviation_above", "spearman_residual_vs_deviation")})

    # ---- figure ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax[0, 0].hist(np.log10(dev_t), bins=50)
    ax[0, 0].set_xlabel("log10 deviation (official vs reference)")
    ax[0, 0].set_title(f"Test split (n={len(dev_t)}): convergence deviation")
    ax[0, 1].loglog(res_t, dev_t, ".", ms=3, label="test samples")
    xs = np.logspace(np.log10(res_t.min()), np.log10(res_t.max()), 50)
    ax[0, 1].loglog(xs, np.exp(icpt) * xs**slope, "r-", label=f"fit: slope {slope:.2f}")
    ax[0, 1].set_xlabel("PDE residual of official target")
    ax[0, 1].set_ylabel("deviation")
    ax[0, 1].legend()
    ax[0, 1].set_title("Residual predicts deviation")
    ax[0, 2].semilogy(frac_t, dev_t, ".", ms=3)
    ax[0, 2].set_xlabel("fraction of cells with a = 1")
    ax[0, 2].set_title("Low permeability -> not converged by t=2")
    g = m.duplicate_groups[0]
    a_g, u_g = read_samples(args.file, g)
    u_ref_g, _ = solve_steady_state(a_g[0], beta, dx)
    dev_g = rel(u_g, np.broadcast_to(u_ref_g, u_g.shape))
    order = np.argsort(dev_g)
    for k, j in enumerate([order[0], order[len(order) // 2], order[-1]]):
        d = u_g[j] - u_ref_g
        v = np.abs(d).max()
        im = ax[1, k].imshow(d.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-v, vmax=v)
        ax[1, k].set_title(f"duplicate a=0.1, sample {g[j]}: official - reference\ndeviation {dev_g[j]:.3f}", fontsize=10)
        fig.colorbar(im, ax=ax[1, k], shrink=0.8)
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=110)
    plt.close(fig)
    rep["figure"] = args.fig
    Path(args.report).write_text(json.dumps(rep, indent=2))
    log_event(log, "report_written", path=args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
