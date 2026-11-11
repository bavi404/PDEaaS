# M1 Audit and Reproducibility Report

Completed 2026-11-11, ≈ 09:10 UTC. No model has been trained. Every number below is copied from a generated report;
the source is given in each table. Headline numbers in DATA.md and PREPROCESSING.md are checked automatically by
`backend/tests/test_docs_traceability.py`.

## 1. Two targets, kept separate

| | Official PDEBench target | Independent steady-state reference |
|---|---|---|
| Storage | `darcy_beta1.0_{train,val,test}.npz` key `u` | `darcy_beta1.0_test_reference.npz` key `u_reference` (test only) |
| Role | training target; **primary benchmark** | secondary analysis only |
| Metrics key | `primary_vs_official_pdebench_target` | `secondary_vs_independent_steady_state_reference` |
| Verified | 1,000/1,000 test targets byte-identical to the raw HDF5 (and train/val, tested) | stored separately; differs from official; not referenced by any training/normalisation code |

## 2. Reference solver verification

Source: `reports/reference_solver_verification.json`, `all_checks_passed: true`.

| # | Check | Result |
|---|---|---|
| 1 | Operator A u = −∇·(a∇u): symmetry / positive definiteness / sign | symmetry error 1.5e-16; min Rayleigh quotient 2,213 > 0; A(point bump) > 0 at the bump |
| 1 | Exact discrete eigenpairs of the Dirichlet 5-point Laplacian (a = 1) | max relative error 4.2e-14 |
| 2 | Boundary diagonals vs closed-form stencil (u ghost = 0, boundary face a = adjacent a) | max relative error 1.7e-16 |
| 2 | Boundary convention vs released data (RMS residual, boundary ring) | fitted 2.9e-3 vs zero-ghost-a 20.6 |
| 3 | Stopping rule | **true** residual ‖b − Ax‖ ≤ tol‖b‖, recomputed before stopping; tol 1e-10, max 20,000 iterations, float64 |
| 3 | Error vs tol, discrete manufactured solution (binary a, 64²) | tol 1e-6 → 2.2e-5; 1e-8 → 2.6e-7; 1e-10 → 9.1e-10; 1e-12 → 1.3e-11 |
| 4 | **All test samples solved** | **1,000 / 1,000**, 8/8 batches converged, max float64 relative residual 9.93e-11, indices aligned with the test split |
| 5 | Residual of the stored references | median 1.7e-4, max 6.2e-4. This is float32 storage rounding: on 5 samples the float64 solutions have residual ≤ 9.9e-11, and casting them to float32 reproduces the stored residuals exactly (stored = float64 to 4.9e-8 relative). Official targets: median 1.4e-3, max 0.45. |
| 6 | Continuum manufactured solution, smooth variable a | rel-L2 6.9e-3, 1.9e-3, 5.0e-4, 1.3e-4 at 16², 32², 64², 128²; observed order 1.86, 1.93, 1.96 |
| 7 | Storage separation | separate files and keys; official test targets byte-identical to raw |
| 8 | No influence on training | only `physics.py` (solver definition) references it; `preprocess.py` does not; normalisers are computed before the reference exists; no reference exists for train/val; guard test `test_reference_isolation.py` |

## 3. Scientific findings

| Finding | Evidence | Decision |
|---|---|---|
| Released data obeys a zero-Dirichlet FV scheme with boundary-face a = adjacent cell a (differs slightly from current public generator code) | boundary residual 2.9e-3 vs 20.6 | Reference solver uses the convention the data obeys |
| **Official targets not fully converged at t = 2** | test split, 1,000 exact solves: deviation median 0.16%, mean 2.93%, p90 8.7%, p99 36.5%, max 55.1%; 29.9% of samples > 1%; driven by low permeability (Spearman −0.949) | Keep official targets unchanged for training and the primary benchmark. Report reference error separately, and stratify by PDE residual. |
| **46 identical inputs (a ≡ 0.1) with different targets** | pairwise difference median 18.9%; 12.5% mean distance to the group mean (irreducible under an L2 loss); 29.9% median deviation from steady state; 45/46 above steady state; residual ranks deviation perfectly (Spearman 1.00) | Group-aware split; all 46 in train at positions 4,414–4,459, outside the 2,000-sample training prefix. They are in no processed file. Revisit if `n_train` > 4,414. |
| **Stride-2 boundary artefact** | interior reconstruction 0.19% vs boundary band 29.3% vs last row/col 93.6% (relative; max abs 0.059). The stride-2 grid covers only [1/256, 253/256]. | Training data is exact at the kept points. 128² evaluation must report interior, boundary band and last row/col separately. |
| 64² input information floor | interior information loss 1.49% mean (1.03% median) | Expected floor for any 64²-input model |
| Resolution | untrained FNO consistent across grids (0.025% / 0.045%) | **Capability only, not generalisation.** Trained-model 64² and 128² errors will be reported separately (PREPROCESSING.md §6). |

## 4. Reproducibility pass

**Command:** `bash scripts/reproduce_m1.sh`, run from `.venv` (Python 3.11.15) and started at 08:32 UTC.

| Stage | Outcome |
|---|---|
| Download step | file present → re-verified size and MD5 against the **live** DaRUS listing: match |
| inspect, preprocess, validate_downsampling, audit, verify_reference_solver | all completed; 8/8 reference batches logged and converged; `all_checks_passed: true` |
| Strict tests (first attempt) | **2 failed, 101 passed**. Both failures were defects in newly written tests, not in the pipeline: (a) a number-format mismatch (`2.9e-03` vs `2.9e-3`); (b) a wrong a-priori bound on the stored-reference residual (`< 1e-4`), refuted by the measurement in §2 row 5. Both tests were corrected to assert the measured property, not loosened arbitrarily. |
| Strict tests (final) | **103 passed, 0 failed, 0 skipped**, with `PDEFLOW_REQUIRE_DATA=1` (`reports/m1_pytest.txt`) |

### Determinism: run 1 (08:13–08:33) vs run 2 (08:32–09:02)

Two independent full runs were compared.

| Artefact | Result |
|---|---|
| `data/splits/darcy_beta1.0_seed20261111.json` | **byte-identical** |
| `data/splits/normalizers_beta1.0_train2000_stride2.json` | **byte-identical** |
| `data_inspection.json` | 0 differing values (4 new keys added in run 2) |
| `preprocessing_summary.json` | 0 differing values |
| `downsampling_validation.json` | 1 difference: CG iteration count 1,595 → 1,596 (new true-residual confirmation). Statistics are identical. |
| `data_audit.json` | 1 difference: wall time 1,038 s → 1,065 s. Statistics are identical. |
| Processed `.npz` arrays | not compared bytewise against run 1 (only file hashes were snapshotted, and those changed because of the new `target_source` label). The arrays are deterministic functions of the byte-identical manifest and the MD5-verified raw file, and the strict tests check them element-wise against the raw HDF5. |

Comparison criterion: numeric values equal to within 1e-9 relative.

### Test inventory (103 tests)

| File | Tests | Data used |
|---|---|---|
| `test_real_data.py` (marker `data`) | 13 | **Real PDEBench file** + processed split files + reference file |
| `test_docs_traceability.py` | 47 | Committed reports generated from the real data |
| `test_data_reader.py` | 11 | **Generated fixture**: tiny synthetic HDF5 mimicking the layout (code paths only) |
| `test_splits_and_normalization.py` | 12 | Generated index lists / random arrays |
| `test_resample_and_physics.py` | 9 | Analytic functions, random binary fields, manufactured solutions |
| `test_metrics.py` | 5 | Generated arrays |
| `test_fno_resolution_support.py` | 4 | Random tensors through an untrained FNO (capability only) |
| `test_reference_isolation.py` | 2 | Static source scan |

Skipped: **0**. In strict mode, a missing real-data artefact fails the test instead of skipping it.

## 5. Corrections made during M1 (for transparency)

1. **Reference-operator sign error** in an early exploratory version (CG solved +∇·(a∇u) = β, giving negative u).
   Caught by the result's sign, then fixed. A test now asserts positivity and the eigenpairs.
2. **Wrong boundary convention:** the first residual analysis used zero-ghost permeability, as in the public code.
   The data contradicted it at the boundary (residual about 21 vs 0.003), and the convention was refitted.
3. **Constant-coefficient test bound:** the first version assumed the unit-square continuum value. Replaced by the
   effective-domain (1 + dx) comparison, which converges at second order.
4. **CG stopping rule:** changed from the recurrence residual to a confirmed true residual.
5. **Provenance timestamp:** re-verification had overwritten `retrieved_utc` with the verification time. It was
   restored from the `download_finished` log event (07:42:50 UTC), and re-verification now writes `last_verified_utc`
   instead.
6. **Logs:** `logs/m1_downsampling.jsonl` from the first downsampling run was deleted before a re-run (superseded);
   all other logs contain every run.
