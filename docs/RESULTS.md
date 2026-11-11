# Results: FNO baseline on PDEBench 2D Darcy Flow (β = 1.0)

Every number below is copied from:
- `reports/m2_eval_fno_baseline.json` (test-set evaluation, `scripts/evaluate.py`)
- `reports/m2_run_fno_baseline/` (training log, history, summary and resolved config)

The test split (1,000 samples) was used **once**, after training finished. It was not used for model selection.

> **Two targets, never mixed** (docs/DATA.md §0)
> - **PRIMARY = official PDEBench target** (`tensor`, unmodified). This is the benchmark the model is trained on and judged against.
> - **SECONDARY = independent steady-state reference** (our CG solve). It is used for analysis only and played no role in training, normalisation or selection.

## 1. Run summary

| Item | Value |
|---|---|
| Model | `neuralop` FNO, 12×12 modes, width 32, 4 layers, **357,217 parameters** |
| Data | 2,000 train / 1,000 val / 1,000 test; trained at **64×64** (stride-2), official targets |
| Optimisation | AdamW, lr 1e-3, weight decay 1e-4, cosine schedule, 150 epochs, batch 16, relative-L2 loss, grad clip 10 |
| Seed | 0 |
| Hardware | 4 vCPU, no GPU; torch 2.15.0+cpu; fp32 |
| Wall time | **5,853 s (97.6 min)** total. Median epoch 36.2 s. Epochs 2–3 took 258 s and 271 s because of CPU contention from a concurrent dry run (docs/TRAINING.md); excluding them, the total is 5,324 s. |
| Peak RSS | **1,076 MB** (training process) |
| Stability | **0 non-finite steps**. Max per-epoch gradient norm 3.08, so the clip at 10 never triggered. |
| Selected checkpoint | **epoch 139**, val rel-L2 (official, 64²) = **0.03028** |
| Checkpoint SHA-256 | `d7133c7c681cf3e9632579833fe357636d84ff7a5678d94b2f24c38955124df7` (`backend/models/fno_darcy_beta1_baseline.pt`) |
| Selection check | Val recomputed from the saved checkpoint gives 0.03028, identical to the selection metric |

Training curve (`docs/figures/m2_fno_baseline_curves.png`):

| epoch | 1 | 10 | 25 | 50 | 100 | 139 | 150 |
|---|---|---|---|---|---|---|---|
| train rel-L2 | 0.361 | 0.055 | 0.036 | 0.023 | 0.0135 | 0.0111 | 0.0110 |
| val rel-L2 (official, 64²) | 0.197 | 0.060 | 0.043 | 0.037 | 0.0313 | 0.0303 | 0.0303 |

The final train/validation gap is 0.011 vs 0.030, which indicates over-fitting with 2,000 training samples. Validation error plateaued after about epoch 110.

## 2. Test error: mean relative L2 over 1,000 samples

| Setting | **PRIMARY vs official** mean | median | p90 | p99 | max | SECONDARY vs reference mean | median |
|---|---|---|---|---|---|---|---|
| **64×64** (training resolution) | **0.0293** | 0.0171 | 0.0640 | 0.1786 | 0.2429 | 0.0391 | 0.0182 |
| **128×128 zero-shot** (native input → native target) | **0.0421** | 0.0315 | 0.0689 | 0.1753 | 0.2441 | 0.0512 | 0.0320 |
| 128×128 via FNO-64 + bilinear interpolation (baseline) | 0.0320 | 0.0205 | 0.0646 | 0.1789 | 0.2432 | 0.0417 | 0.0212 |

For scale, the official target itself differs from the steady-state reference by a mean of 0.0293 and a median of 0.0016 (DATA.md §6).

### Findings
1. **At the training resolution**, mean test error vs the official targets is 2.93%, with a median of 1.71%.
2. **The 128×128 zero-shot evaluation is worse than interpolating the 64×64 prediction:** 4.21% vs 3.20% overall, and 4.02% vs 2.89% in the interior.
   - So this FNO, trained at a single resolution, is **not** discretisation-invariant to within its own error level.
   - Evaluated at 128² and subsampled back to 64², its prediction differs from the 64² prediction by 2.6% on average (`consistency_pred128_subsampled_vs_pred64`).
   - *Implication for the service:* the measured-better path for 128×128 input is to run at 64×64 and interpolate. The exception is the last row and column (§3), where interpolation is extrapolation. M4 will use this result to decide which path to serve.
3. **The model tracks the official targets, not the steady state.** On samples where the official target is far from steady state (PDE residual > 1e-1, n = 80), the 64² error is 8.6% vs official and 16.9% vs the reference. On well-converged samples (residual ≤ 1e-3, n = 450) the two agree at 1.61%. This is expected, because the model is trained on the official targets.
4. **Error depends on how much of the domain is high-permeability.** With a fraction of a=1 cells below 0.2 (mostly a=0.1, large and slowly converging solutions), the error is 8.6% (n = 104). With a fraction between 0.4 and 0.8 it is 1.6–1.9%. The worst test sample (idx 4701; rel-L2 0.244 at 128²) is a nearly all-a=0.1 field; the model under-predicts its peak (`docs/figures/m2_fno_baseline_examples.png`).

## 3. Error by region (mean of per-sample regional relative L2, vs official)

| Region | 64×64 | 128×128 zero-shot | 128×128 interpolation |
|---|---|---|---|
| interior | 0.0289 | 0.0402 | 0.0289 |
| boundary band (2 fine cells) | 0.1197 | 0.3429 | 0.3266 |
| last row + column (128 only) | — | 0.6969 | 0.9679 |

- Boundary-region relative errors are large partly because the target is small near the Dirichlet boundary, which makes the denominator small.
- The last row and column of the 128 grid lies outside the coordinate hull of the stride-2 training grid (PREPROCESSING.md §2), and the 64²-trained model has never seen it.

## 4. Error strata (64×64, vs official; full tables at both resolutions in the report)

| Stratum | n | PRIMARY mean | SECONDARY mean |
|---|---|---|---|
| official-target residual ≤ 1e-3 | 450 | 0.0161 | 0.0161 |
| 1e-3 – 1e-2 | 276 | 0.0249 | 0.0257 |
| 1e-2 – 1e-1 | 194 | 0.0428 | 0.0582 |
| > 1e-1 | 80 | 0.0859 | 0.1689 |
| fraction a=1 in [0, 0.2) | 104 | 0.0864 | 0.1614 |
| [0.2, 0.4) | 203 | 0.0402 | 0.0489 |
| [0.4, 0.6) | 347 | 0.0185 | 0.0194 |
| [0.6, 0.8) | 284 | 0.0155 | 0.0155 |
| [0.8, 1.0] | 62 | 0.0209 | 0.0208 |

## 5. CPU inference latency (4 threads, fp32, model forward only)

| Resolution | Batch | Median ms | p95 ms | ms / sample |
|---|---|---|---|---|
| 64×64 | 1 | 10.6 | 16.5 | 10.6 |
| 64×64 | 32 | 124.4 | 157.2 | 3.9 |
| 128×128 | 1 | 17.6 | 37.3 | 17.6 |
| 128×128 | 32 | 1,483.6 | 1,523.8 | 46.4 |

These were measured on an idle machine after training. Two consecutive evaluation runs differed by up to about 25% (the first run gave 8.3 ms for 64²/batch 1), so treat the numbers as indicative.

## 6. Limitations (stated, not hidden)
- **Training budget:**
  - 150 epochs on 2,000 samples, versus PDEBench's longer schedules and 9,000 samples.
  - This is a single seed, so no variance across seeds is reported.
- **Single training resolution:** 128² zero-shot transfer is measurably imperfect (§2).
- **Not comparable to PDEBench's own runs:**
  - The 46 duplicate-input samples are excluded by construction.
  - The PDEBench split is not used.
  - Errors here are relative L2 on our group-aware split, so they are not directly comparable to PDEBench's published nRMSE numbers.
