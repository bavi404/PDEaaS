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
- [ ] FNO via `build_model(cfg)` (neuraloperator 2.0.0, decided in BLOCKERS §3)
- [ ] Model unit tests (shape, resolution invariance, param count)
- [ ] Training loop: seeds, device=auto, AMP on CUDA, grad clipping, LR schedule
- [ ] Checkpoint save/load (weights + config + normalizer + split hash)
- [ ] Train/val loss log (JSON/CSV) + plot
- [ ] Real training run completed

## M3 — Evaluation
- [ ] Test relative L2 vs OFFICIAL targets (primary) at 64² and 128², separately; interior/boundary; strata
- [ ] Secondary: error vs independent steady-state reference (test only)
- [ ] 128×128 zero-shot evaluation vs interpolated-64² baseline
- [ ] Inference latency benchmark (CPU; batch 1 and batched)
- [ ] `docs/RESULTS.md` generated from report files only

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
