# Training setup (M2 baseline)

- **Config:** `backend/configs/fno_baseline.yaml`.
- **Run artefacts:** `runs/fno_baseline/`, not committed: checkpoints, `train.jsonl`, `history.json`,
  `config.resolved.json`, `summary.json`. Checkpoint SHA-256s are recorded in `reports/`.
- **Results:** `docs/RESULTS.md`, generated from `reports/m2_eval_fno_baseline.json`.

## Decisions

| Item | Choice | Alternatives considered | Justification |
|---|---|---|---|
| Model | `neuralop.models.FNO` 2.0.0, 12×12 modes, width 32, 4 layers; library defaults otherwise (grid positional embedding, lifting/projection ratio 2, channel MLP, soft-gating skip, no domain padding) | in-house FNO2d | Library installs and runs (BLOCKERS §3). Defaults keep the model standard and comparable. Close to the PDEBench Darcy FNO (12 modes, width 20). 357,217 parameters. |
| Training resolution | 64×64, stride-2 | 128×128 | ≈ 8× cheaper per sample (measured), equal to PDEBench `reduced_resolution: 2`. Validated in PREPROCESSING.md §2. |
| Data | train 2,000, validation 1,000; official PDEBench targets | larger N_train | CPU budget. The 2,000-sample prefix holds none of the 46 duplicate inputs. |
| Loss | per-sample relative L2 on **decoded (physical)** u, batch mean | MSE on normalised u | Matches the evaluation metric and PDEBench/FNO practice. It is scale-invariant across samples whose max u spans 0.08–1.2. |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, batch 16 | Adam + step decay (PDEBench) | Standard; decoupled weight decay. |
| Schedule | cosine annealing over 150 epochs | step decay, 500 epochs (PDEBench) | 150 epochs ≈ 85 min on 4 CPUs (measured about 34 s per epoch). Fewer epochs than PDEBench is a stated limitation. |
| Precision | fp32 on CPU; bf16 autocast only on CUDA (none here) | fp16 / AMP on CPU | No GPU in this container. Autocast gives no CPU benefit for FFT layers. |
| Gradient clipping | max-norm 10, logged | none | Safety only. Max gradient norms are logged every epoch; non-finite steps are counted, and the run fails if there are any. |
| Model selection | **lowest validation relative L2 vs official targets at 64×64**, evaluated every epoch | early stopping | The test set and the independent reference are never consulted during training or selection. |
| Reproducibility | seed 0 (Python, NumPy, torch, DataLoader generator); CPU determinism tested | — | `test_training_is_deterministic_on_cpu` |

## What a checkpoint contains (`pdeflow-fno-v1`)

- model `state_dict`
- full resolved config, including SHA-256 of the train/val files, split manifest and normaliser file
- normaliser statistics
- epoch, selection metric value, and parameter count

`pdeflow.training.core.load_checkpoint` rebuilds the model from the stored config. The API (M4) uses exactly this path.

## Run log notes

- **Smoke test** (`--smoke`: 64 train, 64 val, 3 epochs): loss decreased monotonically (train 0.644 → 0.628 → 0.619;
  validation 0.615 → 0.602 → 0.599), with 0 non-finite steps and max gradient norm ≤ 0.46.
- **Baseline run, epochs 2–3:** these took 258 s and 271 s instead of about 35 s, because an evaluation dry run was
  started on the same 4 cores (thread oversubscription). It does not affect results, only the wall-time total, which
  is reported as measured.
- **Baseline run (complete):** started 09:30:05 UTC and ended 11:07:40 UTC on 2026-11-11. Summary:
  - 150 epochs in 5,853 s; median epoch 36.2 s; peak RSS 1,076 MB.
  - 0 non-finite steps. Max pre-clip gradient norm 3.08, so clipping never triggered.
  - Best epoch 139: validation 0.03028.
  - Logs and history are copied to `reports/m2_run_fno_baseline/`; test results are in `docs/RESULTS.md`.
- **Deployable checkpoint:** `backend/models/fno_darcy_beta1_baseline.pt`, a byte-identical copy of `runs/fno_baseline/best.pt`
  (SHA-256 `d7133c7c…124df7`). It is committed because it is small (2.8 MB) and the API must serve exactly the evaluated weights.
- **Test fix:** `test_training_decreases_loss_with_finite_gradients` first ran 25 epochs on the 12-sample toy problem. Its loss
  plateaus there (0.250 → 0.192), which failed the ≥ 2× reduction assertion. The test now runs 60 epochs (measured 0.250 → 0.102)
  and keeps the assertion unchanged.
