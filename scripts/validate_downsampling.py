#!/usr/bin/env python3
"""Validate 128 -> 64 downsampling before any model is trained. Uses TRAIN-split samples only.

Compares stride-2 point subsampling (PDEBench reduced_resolution=2) with 2x2 average pooling:
  1. coordinate consistency with neuraloperator's positional embedding (x_j = j / res)
  2. spectral energy of a and u beyond the 64-grid Nyquist (k > 32) and beyond 12 Fourier modes
  3. target fidelity: relative L2 between u128 and the 64-grid target interpolated back to 128
  4. input information loss: how much the exact solution changes when a128 is replaced by its
     64-grid representation (exact CG solves) -> an irreducible error floor for 64x64 inputs
  5. architecture check (untrained, seeded FNO): does the network give consistent outputs for the
     same field sampled at 64 and 128? smooth band-limited input vs. real binary permeability

Outputs: reports/downsampling_validation.json, docs/figures/downsampling.png, logs/m1_downsampling.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pdeflow.data.dataset import load_split_arrays  # noqa: E402
from pdeflow.data.normalization import DarcyNormalizers  # noqa: E402
from pdeflow.data.physics import solve_steady_state  # noqa: E402
from pdeflow.data.resample import avg_pool, high_frequency_energy_fraction, linear_upsample, subsample  # noqa: E402
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402


def rel_l2(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm((x - ref).reshape(len(x), -1), axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)


def stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def nearest_up(x64: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    j = np.abs(dst[:, None] - src[None, :]).argmin(axis=1)
    return x64[..., j, :][..., :, j]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="data/processed/darcy_beta1.0_train.npz")
    ap.add_argument("--normalizers", default="data/splits/normalizers_beta1.0_train2000_stride2.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="reports/downsampling_validation.json")
    ap.add_argument("--fig", default="docs/figures/downsampling.png")
    ap.add_argument("--log", default="logs/m1_downsampling.jsonl")
    args = ap.parse_args()
    log = get_logger("pdeflow.downsampling", args.log)

    d = load_split_arrays(args.train_file)
    a128, u128 = d["a"][: args.n].astype(np.float64), d["u"][: args.n].astype(np.float64)
    beta = float(d["beta"])
    log_event(log, "loaded", n=args.n, source=args.train_file, split="train")
    R = a128.shape[-1]
    x128 = (np.arange(R) + 0.5) / R
    x_stride = x128[::2]
    x_avg = (np.arange(R // 2) + 0.5) / (R // 2)
    rep: dict = {"n_samples": args.n, "split": "train (prefix)", "beta": beta}

    # 1. coordinates vs embedding ------------------------------------------------------------------
    emb = lambda n: np.arange(n) / n  # noqa: E731  (neuralop regular_grid_2d: linspace(0,1,n+1)[:-1])
    rep["coordinates"] = {
        "offset_true_minus_embedding_128": float(np.unique(np.round(x128 - emb(R), 12))[0]),
        "offset_true_minus_embedding_64_stride": float(np.unique(np.round(x_stride - emb(R // 2), 12))[0]),
        "offset_true_minus_embedding_64_avgpool": float(np.unique(np.round(x_avg - emb(R // 2), 12))[0]),
        "note": "equal offsets at 64 and 128 mean the positional embedding is consistent across resolutions",
        "x128_cell_centres": {"formula": "(2i+1)/256, i=0..127", "first3": x128[:3].tolist(), "last3": x128[-3:].tolist()},
        "x64_stride2": {"formula": "(4j+1)/256, j=0..63  (= x128[::2])", "first3": x_stride[:3].tolist(), "last3": x_stride[-3:].tolist()},
        "x64_avgpool_centres": {"formula": "(4j+2)/256, j=0..63", "first3": x_avg[:3].tolist(), "last3": x_avg[-3:].tolist()},
        "neuralop_embedding_128": {"formula": "i/128", "first3": emb(R)[:3].tolist(), "last3": emb(R)[-3:].tolist()},
        "neuralop_embedding_64": {"formula": "j/64", "first3": emb(R // 2)[:3].tolist(), "last3": emb(R // 2)[-3:].tolist()},
        "stride2_coverage": [float(x_stride[0]), float(x_stride[-1])],
        "fine_cells_outside_stride2_hull": "last row and last column of the 128 grid (x = 255/256)",
    }

    # 2. spectra ---------------------------------------------------------------------------------------
    rep["spectral_energy_fraction"] = {
        "a_above_k32": stats(high_frequency_energy_fraction(a128, 32)),
        "u_above_k32": stats(high_frequency_energy_fraction(u128, 32)),
        "a_above_k12": stats(high_frequency_energy_fraction(a128, 12)),
        "u_above_k12": stats(high_frequency_energy_fraction(u128, 12)),
    }

    # 3. target fidelity --------------------------------------------------------------------------------
    u_s, u_a = subsample(u128, 2), avg_pool(u128, 2)
    rep["target_reconstruction_rel_l2"] = {
        "stride2_then_linear_to_128": stats(rel_l2(linear_upsample(u_s, x_stride, x128), u128)),
        "avgpool_then_linear_to_128": stats(rel_l2(linear_upsample(u_a, x_avg, x128), u128)),
        "stride2_then_linear_to_128_excluding_last_row_and_col": stats(
            rel_l2(linear_upsample(u_s, x_stride, x128)[:, :-1, :-1], u128[:, :-1, :-1])),
        "note": "the stride-2 grid spans x in [1/256, 253/256]; the last fine row/col (255/256) can only be extrapolated",
        "stride2_values_equal_u128_at_kept_points": bool(np.array_equal(u_s, u128[:, ::2, ::2])),
    }
    a_s, a_a = subsample(a128, 2), avg_pool(a128, 2)
    a_s_up, a_a_up = nearest_up(a_s, x_stride, x128), nearest_up(a_a, x_avg, x128)
    rep["input_representation"] = {
        "stride2_unique_values": np.unique(a_s).tolist(),
        "avgpool_unique_values": np.unique(np.round(a_a, 6)).tolist(),
        "stride2_fraction_128_cells_mismatched_after_nearest_up": stats((a_s_up != a128).mean(axis=(1, 2))),
        "avgpool_fraction_coarse_cells_mixed": stats(((a_a > a128.min()) & (a_a < a128.max())).mean(axis=(1, 2))),
    }
    log_event(log, "spectra_and_reconstruction", **rep["spectral_energy_fraction"], **rep["target_reconstruction_rel_l2"])

    # 4. information loss of the 64-grid input (exact solves at 128) ---------------------------------
    dx = 1.0 / R
    us_true, i1 = solve_steady_state(a128, beta, dx)
    us_s, i2 = solve_steady_state(a_s_up, beta, dx)
    us_a, i3 = solve_steady_state(a_a_up, beta, dx)
    rep["input_information_loss_rel_l2"] = {
        "definition": "rel L2 between exact steady states for a128 and for its 64-grid representation (nearest-upsampled)",
        "stride2": stats(rel_l2(us_s, us_true)), "avgpool_arithmetic": stats(rel_l2(us_a, us_true)),
        "cg": [i1, i2, i3],
    }
    log_event(log, "information_loss", **rep["input_information_loss_rel_l2"])

    # 4b. interior vs boundary breakdown on the 128 grid ------------------------------------------------
    dist = np.minimum.outer(np.minimum(np.arange(R), np.arange(R)[::-1]), np.minimum(np.arange(R), np.arange(R)[::-1]))
    regions = {"interior (>=2 fine cells from edge)": dist >= 2, "boundary band (outer 2 fine cells)": dist < 2,
               "last fine row+col only (extrapolated by stride-2)": np.zeros((R, R), bool)}
    regions["last fine row+col only (extrapolated by stride-2)"][-1, :] = True
    regions["last fine row+col only (extrapolated by stride-2)"][:, -1] = True

    def region_rel(x, ref, mask):
        return np.linalg.norm((x - ref)[:, mask], axis=1) / np.linalg.norm(ref[:, mask], axis=1)

    up_s, up_a = linear_upsample(u_s, x_stride, x128), linear_upsample(u_a, x_avg, x128)
    rep["region_breakdown"] = {"definition": "rel-L2 restricted to region: ||(x-ref)[region]|| / ||ref[region]||; max_abs in units of u",
                               "kept_points_max_abs_discrepancy_stride2": float(np.abs(u_s - u128[:, ::2, ::2]).max())}
    for name, mask in regions.items():
        rep["region_breakdown"][name] = {
            "n_cells": int(mask.sum()),
            "target_reconstruction_stride2": stats(region_rel(up_s, u128, mask)),
            "target_reconstruction_stride2_max_abs": float(np.abs((up_s - u128)[:, mask]).max()),
            "target_reconstruction_avgpool": stats(region_rel(up_a, u128, mask)),
            "target_reconstruction_avgpool_max_abs": float(np.abs((up_a - u128)[:, mask]).max()),
            "input_mismatch_fraction_stride2": stats((a_s_up != a128)[:, mask].mean(axis=1)),
            "information_loss_stride2": stats(region_rel(us_s, us_true, mask)),
            "information_loss_avgpool": stats(region_rel(us_a, us_true, mask)),
        }
    log_event(log, "region_breakdown", **rep["region_breakdown"])

    # 5. architecture discretisation consistency (untrained FNO) -------------------------------------
    from neuralop.models import FNO

    torch.manual_seed(args.seed)
    model = FNO(n_modes=(12, 12), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4).eval()
    norms = DarcyNormalizers.from_json(Path(args.normalizers).read_text())
    X, Y = np.meshgrid(x128, x128, indexing="ij")
    rng = np.random.default_rng(args.seed)
    smooth = np.zeros((8, R, R))
    for k in range(8):
        for _ in range(4):
            kx, ky = rng.integers(1, 5, size=2)
            smooth[k] += rng.normal() * np.sin(np.pi * kx * X) * np.sin(np.pi * ky * Y)
    with torch.no_grad():
        def run(x):
            return model(torch.tensor(x, dtype=torch.float32)[:, None]).numpy()[:, 0]
        sm128, sm64 = run(smooth), run(smooth[:, ::2, ::2])
        ea = norms.a.encode(a128[:16])
        ar128, ar64 = run(ea), run(ea[:, ::2, ::2])
    rep["architecture_consistency_untrained_fno"] = {
        "config": "neuralop FNO n_modes=(12,12) hidden=32 layers=4, untrained, seed %d" % args.seed,
        "metric": "rel L2 between model(x64) and model(x128)[::2,::2]",
        "smooth_bandlimited_input": stats(rel_l2(sm64, sm128[:, ::2, ::2])),
        "binary_permeability_input": stats(rel_l2(ar64, ar128[:, ::2, ::2])),
    }
    log_event(log, "architecture_consistency", **rep["architecture_consistency_untrained_fno"])

    # figure ----------------------------------------------------------------------------------------------
    i = int(np.argmax(high_frequency_energy_fraction(a128, 32)))
    fig, ax = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    panels = [(a128[i], "a 128x128"), (a_s[i], "a 64x64 (stride 2)"), (a_a[i], "a 64x64 (2x2 avg pool)"),
              (np.abs(a_s_up[i] - a128[i]), "|a128 - nearest_up(a64 stride)|"),
              (u128[i], "u 128x128"), (u_s[i], "u 64x64 (stride 2)"),
              (np.abs(linear_upsample(u_s[i:i + 1], x_stride, x128)[0] - u128[i]), "|u128 - lin_up(u64 stride)|"),
              (np.abs(us_s[i] - us_true[i]), "|u_ss(a128) - u_ss(a64 stride)|")]
    for k, (img, title) in enumerate(panels):
        axk = ax.flat[k]
        im = axk.imshow(img.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
        axk.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axk, shrink=0.8)
    fig.suptitle(f"Downsampling validation, train sample with the most high-frequency energy in a (train idx {int(d['indices'][i])})")
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=110)
    plt.close(fig)
    rep["figure"] = args.fig

    Path(args.report).write_text(json.dumps(rep, indent=2))
    log_event(log, "report_written", path=args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
