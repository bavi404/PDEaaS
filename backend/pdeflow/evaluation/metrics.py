"""Evaluation metrics with two strictly separated targets.

PRIMARY benchmark   : error against the OFFICIAL PDEBench target (`tensor`, unmodified). This is what the
                      FNO is trained to reproduce and the only number reported as "test error".
SECONDARY analysis  : error against the independent exact steady-state REFERENCE (CG solve computed by this
                      project, docs/DATA.md §4). Used only to interpret results; never used for training,
                      model selection or early stopping.

The two are returned under different top-level keys and can never be merged by accident: the reference
is an optional, separately named argument.
"""

from __future__ import annotations

import numpy as np

PRIMARY = "primary_vs_official_pdebench_target"
SECONDARY = "secondary_vs_independent_steady_state_reference"


def relative_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample ||pred - target||_2 / ||target||_2 over the spatial axes (last two)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    n = pred.shape[0]
    den = np.linalg.norm(target.reshape(n, -1), axis=1)
    if np.any(den == 0):
        raise ValueError("target with zero norm")
    return np.linalg.norm((pred - target).reshape(n, -1), axis=1) / den


def summarize(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=np.float64)
    return {"n": int(err.size), "mean": float(err.mean()), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)), "p99": float(np.percentile(err, 99)), "max": float(err.max())}


def evaluate(pred: np.ndarray, official: np.ndarray, reference: np.ndarray | None = None,
             strata: dict[str, np.ndarray] | None = None) -> dict:
    """Return {PRIMARY: ..., SECONDARY: ...}. ``strata`` maps a name to a boolean mask over samples."""
    out: dict = {}
    e_off = relative_l2(pred, official)
    out[PRIMARY] = {"relative_l2": summarize(e_off)}
    if strata:
        out[PRIMARY]["strata"] = {k: summarize(e_off[m]) for k, m in strata.items() if np.any(m)}
    if reference is not None:
        e_ref = relative_l2(pred, reference)
        out[SECONDARY] = {"relative_l2": summarize(e_ref),
                          "official_target_vs_reference": summarize(relative_l2(official, reference))}
        if strata:
            out[SECONDARY]["strata"] = {k: summarize(e_ref[m]) for k, m in strata.items() if np.any(m)}
    return out


def region_masks(n: int, band: int) -> dict[str, np.ndarray]:
    """Boolean (n, n) masks: interior (>= ``band`` cells from every edge), boundary band, and last row+col.

    At 128x128 use band=2 (two fine cells); at 64x64 use band=1 (one coarse cell = two fine cells).
    The last row+col of the 128 grid lies outside the stride-2 training grid's coordinate hull.
    """
    idx = np.arange(n)
    edge = np.minimum(idx, idx[::-1])
    dist = np.minimum.outer(edge, edge)
    last = np.zeros((n, n), bool)
    last[-1, :] = True
    last[:, -1] = True
    return {"interior": dist >= band, "boundary_band": dist < band, "last_row_col": last}


def regional_relative_l2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 restricted to ``mask``: ||(pred-target)[mask]|| / ||target[mask]||."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.linalg.norm((pred - target)[:, mask], axis=1) / np.linalg.norm(target[:, mask], axis=1)
