# PDEFlow Task Checklist

Legend: `[x]` done and verified · `[ ]` open · `[!]` blocked (reason given)

## M0 — Discovery & planning
- [x] Inspect working dir, git, Python, packages, disk, memory, GPU/CUDA, Node, Docker (`docs/ENVIRONMENT.md`)
- [x] Architecture, decisions, risks, milestones (`docs/PLAN.md`)
- [x] Repository skeleton + git init
- [x] Blocker assessment after network-policy change (`docs/BLOCKERS.md`)
- [!] Push to GitHub — GH token invalid in this environment (needs user, see BLOCKERS §6.2)

## M1 — Data ✅ (audit + reproducibility: `docs/M1_AUDIT_REPORT.md`)
- [x] Project venv (`.venv`, Python 3.11.15) + pinned `backend/requirements.lock`
- [x] Full β=1.0 file downloaded (1,310,724,488 B); MD5 matches live DaRUS listing; provenance with SHA-256
- [x] HDF5 structure, sample count, shapes, dtypes, ranges → `docs/DATA.md`
- [x] PDE / forcing / BC verified against data; boundary-face convention fitted (residual 2.9e-3 vs 20.6)
- [x] Independent steady-state reference solver verified (8 checks, incl. eigenpairs + 2nd-order MMS)
- [x] Reference for all 1,000 test samples solved and converged (8/8 batches), stored separately from official targets
- [x] Convergence audit (test deviation median 0.16%, mean 2.93%) and duplicate audit (46 × a≡0.1); decisions documented
- [x] Group-aware seeded 80/10/10 split (8000/1000/1000); duplicates in no processed file
- [x] Train-only scalar normalisers
- [x] 128→64 stride-2 validated: exact coordinates, interior vs boundary errors, info-loss floor
- [x] Evaluation plan fixed before training: 64² and 128² separately, primary = official, secondary = reference
- [x] Visualisations: samples, histograms, downsampling, convergence audit
- [x] `scripts/reproduce_m1.sh` full re-run; run1 vs run2 deterministic; strict tests 103 passed / 0 skipped
## M2 — Model & training
- [x] FNO via `build_model(cfg)` (neuraloperator 2.0.0, decided in BLOCKERS §3)
- [x] Model unit tests (shape at 64² and 128², param count 357,217, loss = metric, determinism, checkpoint round-trip)
- [x] Training loop: seeds, device=auto, bf16 autocast on CUDA only, grad clipping, cosine LR schedule
- [x] Checkpoint save/load (weights + config + normalizers + data/split/normaliser SHA-256)
- [x] Smoke test: loss decreasing, finite gradients
- [x] Train/val log (JSON lines + history.json) + curve plot
- [x] Real training run completed: 150 epochs, 97.6 min CPU, best val 0.0303 (epoch 139), 0 non-finite steps
- [x] Deployable checkpoint committed with SHA-256 (`backend/models/fno_darcy_beta1_baseline.pt`)

## M3 — Evaluation
- [x] Test relative L2 vs OFFICIAL targets (primary) at 64² (0.0293) and 128² (0.0421), separately; regions; strata
- [x] Secondary: error vs independent steady-state reference (test only): 0.0391 / 0.0512
- [x] 128×128 zero-shot evaluation vs interpolated-64² baseline (interpolation better: 0.0320)
- [x] Inference latency benchmark (CPU; batch 1 and 32)
- [x] `docs/RESULTS.md` generated from report files only

## M4 — Backend
- [ ] FastAPI: /health, /model, /samples, /samples/{id}, /predict
- [ ] Input validation (shape, finite values, positive permeability, size limits)
- [ ] Structured JSON logs with request IDs and latency
- [ ] Pytest API tests

## M5 — Frontend
- [ ] Next.js + TS app with Plotly heatmaps (input, prediction, truth, error)
- [ ] Sample picker + editable/random coefficient field
- [ ] Error/loading states
- [ ] Vitest + RTL tests; production build

## M6 — Deployment
- [x] Docker pull/run verified; build-time network needs proxy args (BLOCKERS §2.4)
- [ ] Dockerfiles (api, web) + docker-compose.yml with health checks
- [ ] Compose verified end-to-end
- [ ] Public deployment live and verified

## M7 — Documentation & submission
- [ ] README (quickstart, results, limitations)
- [ ] `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/RESULTS.md`, API contract
- [ ] Clean git history
- [ ] Final submission checklist reflects reality
