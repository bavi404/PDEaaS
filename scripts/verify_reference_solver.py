#!/usr/bin/env python3
"""Verify the independent steady-state REFERENCE solver and the stored test-split references.

Checks (all numbers saved to reports/reference_solver_verification.json):
  1  operator + sign: symmetry, positive definiteness, exact discrete eigenpairs for a = 1
  2  boundary treatment: diagonal entries at boundary cells vs. the closed-form stencil, incl. the
     boundary-face permeability (= adjacent cell's a); data-fit evidence quoted from data_inspection.json
  3  convergence criterion: true-residual stopping rule; error vs tol on a discrete manufactured solution
  4  all stored test references: count, alignment, convergence log of the run that produced them
  5  residuals of the stored references (float32 storage) recomputed from scratch
  6  continuum manufactured solution with variable smooth a: second-order convergence 16 -> 128
  7  storage separation from the official PDEBench targets
  8  no influence on training / normalisation / model selection (code scan + artefact provenance order)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
from pdeflow.data.dataset import load_split_arrays  # noqa: E402
from pdeflow.data.pdebench import read_samples  # noqa: E402
from pdeflow.data.physics import CG_DEFAULTS, apply_operator, fv_residual, solve_linear  # noqa: E402
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402


def stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/reference_solver_verification.json")
    ap.add_argument("--log", default="logs/m1_reference_verification.jsonl")
    args = ap.parse_args()
    log = get_logger("pdeflow.refverify", args.log)
    rng = np.random.default_rng(0)
    rep: dict = {"solver_configuration": CG_DEFAULTS}
    checks: dict = {}

    # 1. operator, sign ---------------------------------------------------------------------------------
    n, h = 32, 1 / 32
    a = np.where(rng.random((n, n)) > 0.5, 1.0, 0.1)
    v, w = rng.normal(size=(2, n, n))
    sym = abs(np.sum(v * apply_operator(a, w, h)) - np.sum(w * apply_operator(a, v, h))) / abs(np.sum(v * apply_operator(a, w, h)))
    rayleigh = [float(np.sum(z * apply_operator(a, z, h)) / np.sum(z * z)) for z in rng.normal(size=(20, n, n))]
    # a = 1: the stencil is the standard Dirichlet 5-point Laplacian on an (n+2)-node line with zero ends
    i = np.arange(1, n + 1)
    eig_err = []
    for j, k in [(1, 1), (2, 3), (n, n), (5, 17)]:
        phi = np.outer(np.sin(j * np.pi * i / (n + 1)), np.sin(k * np.pi * i / (n + 1)))
        lam = 4 / h**2 * (np.sin(j * np.pi / (2 * (n + 1))) ** 2 + np.sin(k * np.pi / (2 * (n + 1))) ** 2)
        eig_err.append(float(np.abs(apply_operator(np.ones((n, n)), phi, h) - lam * phi).max() / (lam * np.abs(phi).max())))
    bump = np.zeros((n, n))
    bump[n // 2, n // 2] = 1.0
    checks["1_operator_and_sign"] = {
        "definition": "A u = -div(a grad u); fv_residual(a,u,beta) = beta - A u",
        "symmetry_rel_error": float(sym), "min_rayleigh_quotient_random_vectors": min(rayleigh),
        "positive_definite": bool(min(rayleigh) > 0),
        "A(point bump) at bump > 0 (correct sign of -div grad)": bool(apply_operator(a, bump, h)[n // 2, n // 2] > 0),
        "exact_discrete_eigenpairs_a1_max_rel_error": max(eig_err),
        "fv_residual_identity_max_err": float(np.abs(fv_residual(a, v, 1.0, h) - (1.0 - apply_operator(a, v, h))).max()),
    }

    # 2. boundary treatment ------------------------------------------------------------------------------
    a2 = rng.choice([0.1, 1.0], size=(n, n))
    diag_err = 0.0
    for (p, q) in [(0, 0), (0, 7), (n - 1, 3), (11, n - 1), (n - 1, n - 1), (9, 13)]:
        e = np.zeros((n, n))
        e[p, q] = 1.0
        got = apply_operator(a2, e, h)[p, q]
        ag = np.pad(a2, 1, mode="edge")  # fitted convention: boundary-face a = adjacent cell a
        P, Q = p + 1, q + 1
        faces = [0.5 * (ag[P, Q] + ag[P + dp, Q + dq]) for dp, dq in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        diag_err = max(diag_err, abs(got - sum(faces) / h**2) / (sum(faces) / h**2))
    insp = json.loads((REPO / "reports/data_inspection.json").read_text())["pde_residual"]
    checks["2_boundary"] = {
        "u_condition": "u = 0 at ghost-cell centres (half a cell outside [0,1]); effective domain side 1 + dx",
        "boundary_face_permeability": "equal to the adjacent interior cell (fitted to released data)",
        "diagonal_vs_closed_form_max_rel_error": float(diag_err),
        "data_fit_boundary_ring_rms_residual": {"fitted_edge_convention": insp["first200_rms_boundary_ring"],
                                                "zero_ghost_permeability_(public_code)": insp.get("first200_rms_boundary_ring_if_a_ghost_zero")},
    }

    # 3. convergence criterion: discrete manufactured solution (exact answer known to round-off) ------
    a3 = np.where(rng.random((4, 64, 64)) > 0.5, 1.0, 0.1)
    u_star = rng.random((4, 64, 64))
    b3 = apply_operator(a3, u_star, 1 / 64)
    by_tol = {}
    for tol in (1e-6, 1e-8, 1e-10, 1e-12):
        x, info = solve_linear(a3, b3, 1 / 64, tol=tol)
        by_tol[str(tol)] = {"converged": info["converged"], "iterations": info["iterations"], "true_rel_residual": info["max_relative_residual"],
                            "max_rel_error_vs_exact": float((np.linalg.norm((x - u_star).reshape(4, -1), axis=1) / np.linalg.norm(u_star.reshape(4, -1), axis=1)).max())}
    checks["3_convergence_criterion"] = {"stopping_rule": CG_DEFAULTS["stopping_rule"] + " (TRUE residual recomputed before stopping)",
                                        "discrete_manufactured_solution_64x64_binary_a": by_tol}

    # 4 + 5. stored test references -------------------------------------------------------------------------
    ref = np.load(REPO / "data/processed/darcy_beta1.0_test_reference.npz")
    test = load_split_arrays(REPO / "data/processed/darcy_beta1.0_test.npz")
    u_ref = ref["u_reference"]
    audit = json.loads((REPO / "reports/data_audit.json").read_text())
    batches = audit["convergence_test_split"]["cg_batches"]
    res_ref = np.sqrt(((1.0 - apply_operator(test["a"], u_ref.astype(np.float64), 1 / 128)) ** 2).sum(axis=(1, 2))) / 128.0
    res_off = np.sqrt(((1.0 - apply_operator(test["a"], test["u"].astype(np.float64), 1 / 128)) ** 2).sum(axis=(1, 2))) / 128.0
    checks["4_all_test_samples"] = {
        "n_reference": int(len(u_ref)), "n_test": int(len(test["indices"])),
        "indices_aligned": bool(np.array_equal(ref["indices"], test["indices"])),
        "batches_logged": len(batches), "samples_reported_by_audit": int(audit["convergence_test_split"]["n"]),
        "all_batches_converged": bool(all(b["converged"] for b in batches)),
        "max_solver_reported_rel_residual_float64": max(b["max_relative_residual"] for b in batches),
        "finite_and_positive": bool(np.isfinite(u_ref).all() and (u_ref > 0).all()),
    }
    checks["5_residual_of_stored_solutions"] = {
        "definition": "||beta - A u||_2 / ||beta||_2 recomputed in float64 from the STORED float32 arrays",
        "reference_float32_stored": stats(res_ref),
        "official_targets_for_comparison": stats(res_off),
        "note": "float32 storage rounding bounds the stored-reference residual well above the 1e-10 float64 solve",
    }

    # 6. continuum manufactured solution, variable smooth a --------------------------------------------------
    def mms_error(N: int) -> float:
        hh = 1.0 / N
        x = (np.arange(N) + 0.5) * hh
        X, Y = np.meshgrid(x, x, indexing="ij")
        L = 1.0 + hh
        p = np.pi / L
        s, t = X + hh / 2, Y + hh / 2  # u* = 0 exactly at the ghost-cell centres
        u = np.sin(p * s) * np.sin(p * t)
        ux, uy = p * np.cos(p * s) * np.sin(p * t), p * np.sin(p * s) * np.cos(p * t)
        A = 1 + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        Ax, Ay = np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.pi * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        f = -(Ax * ux + Ay * uy) + A * 2 * p**2 * u
        uh, info = solve_linear(A, f, hh, tol=1e-12)
        assert info["converged"]
        return float(np.linalg.norm(uh - u) / np.linalg.norm(u))
    errs = {N: mms_error(N) for N in (16, 32, 64, 128)}
    Ns = sorted(errs)
    checks["6_manufactured_solution_continuum"] = {
        "u_star": "sin(pi s/L) sin(pi t/L), s=x+dx/2, t=y+dx/2, L=1+dx (zero at ghost centres)",
        "a": "1 + 0.5 sin(2 pi x) cos(2 pi y)", "f": "-div(a grad u_star), analytic",
        "rel_l2_error": {str(k): v for k, v in errs.items()},
        "observed_order": {f"{a_}->{b_}": float(np.log2(errs[a_] / errs[b_])) for a_, b_ in zip(Ns[:-1], Ns[1:])},
    }

    # 7. storage separation -------------------------------------------------------------------------------
    _, u_raw = read_samples(REPO / "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5", test["indices"])
    checks["7_storage_separation"] = {
        "official_file": "data/processed/darcy_beta1.0_test.npz (key 'u')",
        "reference_file": "data/processed/darcy_beta1.0_test_reference.npz (key 'u_reference')",
        "official_test_targets_byte_identical_to_raw_hdf5": bool(np.array_equal(test["u"], u_raw)),
        "reference_differs_from_official": bool(not np.array_equal(u_ref, test["u"])),
        "reference_keys": sorted(ref.files), "official_keys": sorted(test.keys()),
    }

    # 8. no influence on training / normalisation / model selection -----------------------------------------
    hits = {}
    for path in sorted((REPO / "backend/pdeflow").rglob("*.py")) + [REPO / "scripts/preprocess.py"]:
        txt = path.read_text()
        found = [tok for tok in ("u_reference", "test_reference", "solve_steady_state") if tok in txt]
        if found:
            hits[str(path.relative_to(REPO))] = found
    allowed = {"backend/pdeflow/data/physics.py", "backend/pdeflow/evaluation/metrics.py"}  # solver + secondary metric
    norm_path = REPO / "data/splits/normalizers_beta1.0_train2000_stride2.json"
    checks["8_no_influence_on_training"] = {
        "files_referencing_reference_or_solver": hits,
        "only_solver_definition_references_it": bool(set(hits) <= allowed),
        "preprocess_py_reads_reference": "scripts/preprocess.py" in hits,
        "normalizer_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "pipeline_order": "preprocess.py (splits + normalisers) runs BEFORE audit_data.py creates the reference",
        "reference_split": "test only; there is no reference for train or val, so it cannot enter training or model selection",
    }

    rep["checks"] = checks
    ok = (checks["1_operator_and_sign"]["positive_definite"] and checks["1_operator_and_sign"]["exact_discrete_eigenpairs_a1_max_rel_error"] < 1e-10
          and checks["2_boundary"]["diagonal_vs_closed_form_max_rel_error"] < 1e-12
          and checks["3_convergence_criterion"]["discrete_manufactured_solution_64x64_binary_a"]["1e-10"]["converged"]
          and checks["4_all_test_samples"]["n_reference"] == 1000 and checks["4_all_test_samples"]["all_batches_converged"]
          and checks["4_all_test_samples"]["indices_aligned"]
          and min(checks["6_manufactured_solution_continuum"]["observed_order"].values()) > 1.8
          and checks["7_storage_separation"]["official_test_targets_byte_identical_to_raw_hdf5"]
          and checks["8_no_influence_on_training"]["only_solver_definition_references_it"])
    rep["all_checks_passed"] = bool(ok)
    Path(args.report).write_text(json.dumps(rep, indent=2))
    log_event(log, "reference_solver_verified", all_checks_passed=bool(ok), report=args.report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
