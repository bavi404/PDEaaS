"""Discrete Darcy operator matching the PDEBench generator's finite-volume stencil.

PDEBench generates Darcy data by integrating  du/dt = div(a grad u) + beta  to t = 2 on a
cell-centred grid with zero ghost values for u (a zero-Dirichlet condition imposed at the
ghost-cell centre, half a cell outside the domain; the generator calls this mode "Neumann").
Interior face coefficients are arithmetic means of the two adjacent cells.

Boundary-face coefficient: fitted against the *released* file (docs/DATA.md). The stored data
satisfies the stencil to ~3e-3 at the boundary when the boundary face uses the adjacent cell's
a ("edge" ghost for a), versus ~2e1 with a zero ghost for a as in the current public generator
code. We therefore use the edge convention, which is what the released data actually obeys.

The residual ``r = beta + div(a grad u)`` equals du/dt of that scheme, so |r| ~ 0 means the
stored field is a steady state of the discrete equation  -div(a grad u) = beta.
"""

from __future__ import annotations

import numpy as np


def fv_residual(a: np.ndarray, u: np.ndarray, beta: float, dx: float, dy: float | None = None,
                a_ghost: str = "edge") -> np.ndarray:
    """Finite-volume residual for arrays of shape (..., Nx, Ny). Computed in float64.

    ``a_ghost``: "edge" (fitted to the released data, default) or "zero" (current public generator code).
    """
    if a_ghost not in ("edge", "zero"):
        raise ValueError("a_ghost must be 'edge' or 'zero'")
    dy = dx if dy is None else dy
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    pad = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    ag = np.pad(a, pad, mode="edge") if a_ghost == "edge" else np.pad(a, pad)  # see module doc
    ug = np.pad(u, pad)  # u ghost = 0  (zero Dirichlet)
    # x-faces: between cells i and i+1 for i = -1..Nx-1  -> shape (..., Nx+1, Ny)
    ax = 0.5 * (ag[..., 1:, 1:-1] + ag[..., :-1, 1:-1])
    fx = -ax * (ug[..., 1:, 1:-1] - ug[..., :-1, 1:-1]) / dx
    ay = 0.5 * (ag[..., 1:-1, 1:] + ag[..., 1:-1, :-1])
    fy = -ay * (ug[..., 1:-1, 1:] - ug[..., 1:-1, :-1]) / dy
    div_flux = (fx[..., 1:, :] - fx[..., :-1, :]) / dx + (fy[..., :, 1:] - fy[..., :, :-1]) / dy
    return beta - div_flux


def relative_residual(a: np.ndarray, u: np.ndarray, beta: float, dx: float) -> np.ndarray:
    """Per-sample RMS residual divided by |beta| (dimensionless; 0 = exact discrete steady state)."""
    r = fv_residual(a, u, beta, dx)
    return np.sqrt((r**2).mean(axis=(-2, -1))) / abs(beta)


def apply_operator(a: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    """A u = -div(a grad u) with the same stencil/boundary as ``fv_residual`` (SPD for a > 0)."""
    return -fv_residual(a, u, 0.0, dx)  # fv_residual(beta=0) = +div(a grad u)


CG_DEFAULTS = {"method": "conjugate gradient (unpreconditioned), batched over samples",
               "operator": "A u = -div(a grad u); 5-point finite-volume stencil, arithmetic face means",
               "boundary": "u ghost = 0 (zero Dirichlet at ghost-cell centre); boundary-face a = adjacent cell a",
               "stopping_rule": "||b - A x||_2 <= tol * ||b||_2 for every sample in the batch",
               "tol": 1e-10, "max_iter": 20000, "dtype": "float64", "initial_guess": "zeros"}


def solve_linear(a: np.ndarray, rhs: np.ndarray, dx: float, tol: float = 1e-10, max_iter: int = 20000) -> tuple[np.ndarray, dict]:
    """Batched CG for A u = rhs with A = -div(a grad u) (SPD for a > 0). Shapes (B, n, n) or (n, n)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    single = a.ndim == 2
    if single:
        a, b = a[None], b[None]
    if a.shape != b.shape:
        raise ValueError(f"a and rhs shapes differ: {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and (a > 0).all()):
        raise ValueError("permeability must be finite and strictly positive")
    if not np.isfinite(b).all():
        raise ValueError("rhs must be finite")
    x = np.zeros_like(a)
    r = b.copy()
    p = r.copy()
    rs = (r * r).sum(axis=(1, 2))
    b_norm = np.sqrt((b * b).sum(axis=(1, 2)))
    b_norm = np.where(b_norm == 0, 1.0, b_norm)
    it = 0
    true_rel = np.full(a.shape[0], np.inf)
    for it in range(1, max_iter + 1):
        Ap = apply_operator(a, p, dx)
        alpha = rs / (p * Ap).sum(axis=(1, 2))
        x += alpha[:, None, None] * p
        r -= alpha[:, None, None] * Ap
        rs_new = (r * r).sum(axis=(1, 2))
        if np.all(np.sqrt(rs_new) <= tol * b_norm):
            # confirm with the TRUE residual; if recurrence drift left it above tol, restart from it
            r = b - apply_operator(a, x, dx)
            rs_new = (r * r).sum(axis=(1, 2))
            true_rel = np.sqrt(rs_new) / b_norm
            if np.all(true_rel <= tol):
                break
            p = r.copy()
            rs = rs_new
            continue
        p = r + (rs_new / rs)[:, None, None] * p
        rs = rs_new
    true_rel = np.sqrt(((b - apply_operator(a, x, dx)) ** 2).sum(axis=(1, 2))) / b_norm
    info = {"iterations": it, "max_relative_residual": float(true_rel.max()), "converged": bool(true_rel.max() <= tol),
            "tol": tol, "max_iter": max_iter}
    return (x[0] if single else x), info


def solve_steady_state(a: np.ndarray, beta: float, dx: float, tol: float = 1e-10, max_iter: int = 20000) -> tuple[np.ndarray, dict]:
    """Exact discrete steady state of -div(a grad u) = beta (the independent REFERENCE, docs/DATA.md §4).

    Returns (u, info) with u in float64. Used for data validation and as a secondary evaluation target;
    never as a training target.
    """
    a = np.asarray(a, dtype=np.float64)
    return solve_linear(a, np.full_like(a, beta), dx, tol=tol, max_iter=max_iter)
