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
- [x] Serves the evaluated checkpoint: SHA-256 pinned, startup fails on mismatch; full test split through the API path == report (bit-exact)
- [x] FastAPI: /health, /model, /samples, /samples/{id}, /predict (sample or user field; 64² / 128²; strategies)
- [x] Official target vs independent reference kept separate in samples, metrics and responses
- [x] Input validation (shape, finite, positive, bounds, body size) + distribution-shift warnings
- [x] Structured JSON logs with request IDs and latency; generic 500 with logged traceback
- [x] Pytest API tests (36) + HTTP smoke test with uvicorn; latency report
- [x] Bundled 24 CC BY 4.0 test samples with attribution
- [x] docs/API.md; HF Spaces configuration and secrets plan in docs/DEPLOYMENT.md (not deployed)

## M5 — Frontend
- [x] Next.js 16 + TypeScript + Plotly; pinned dependencies; same-origin proxy to FastAPI (runtime `PDEFLOW_API_URL`)
- [x] Dashboard (intro, PDE, model overview, performance, recent activity via new `GET /activity`)
- [x] Solver workspace (Darcy, benchmark examples, custom generator / paste / upload, 64²/128², inference path, input heatmap)
- [x] Results (benchmark: input / prediction / official target / |error| + separate secondary reference section; custom: no ground truth, no metrics)
- [x] Model information (architecture, training, evaluation, limitations, provenance), all from `/model`
- [x] Loading / success / empty / error states; OOD warnings; interactive heatmaps with hover, axes, colour bars
- [x] Vitest + RTL: 52 tests; Playwright on the real stack: 26 tests (desktop + mobile, real unavailable backend, console-error checks)
- [x] Screenshots (`docs/screenshots/`), `docs/FRONTEND.md`

## M6 — Docker and deployment preparation
- [x] Docker pull/run verified; build-time network needs proxy args (BLOCKERS §2.4)
- [x] Multi-stage Dockerfile (Node build → Python venv → slim runtime); Python 3.11.18; base images pinned by digest
- [x] Runtime lock constrained to the tested dev lock; `pip check` clean; no dataset, runs, tests or credentials in the image
- [x] Single container: Next.js on 7860 → FastAPI on 127.0.0.1:8000; UID 1000; HEALTHCHECK; graceful SIGTERM; crash → exit
- [x] Checkpoint SHA-256 verified at build, at entrypoint start and on API load (tampering → exit 3)
- [x] Container tests: 17/17 functional checks, 26/26 Playwright e2e against the container, lifecycle 4/4
- [x] Resources within HF CPU basic: peak 400–434 MiB, 2 vCPU, startup about 8.5 s
- [x] docker-compose.yml (local), docs/DOCKER.md, docs/DEPLOYMENT.md, Space card, deploy script (dry-run only)
- [ ] ~~Public Docker Space deployment~~: superseded. HF Docker Spaces need a paid plan; see M7 (ZeroGPU Gradio Space)
- [ ] Public URL verified (health SHA, functional checks, e2e)

## M7 — Zero-cost hosting (ZeroGPU Gradio Space)
- [x] Feasibility: official free-tier terms; Docker/Gradio Spaces need PRO; ZeroGPU Gradio free for eligible accounts
- [x] Owner confirmed eligibility (verified email, account > 30 days)
- [x] `deploy/zerogpu/`: app (FastAPI /api + Gradio /gradio + static frontend /), Python 3.12 lock, Space card, tests
- [x] Gradio ↔ FastAPI predictions identical (SHA-256 of float64 arrays); checkpoint SHA unchanged
- [x] Strict backend suites 3.11 and 3.12 (151 + 151), Space tests 31, 1,000-sample equivalence bit-identical (3 envs)
- [x] Frontend typecheck, Vitest 52, standalone + static builds, Playwright: integrated 24 + Gradio 6, regression 26
- [ ] **Public deployment**: two ZeroGPU attempts failed (build fixed; runtime requires `@spaces.GPU`); Render Free prepared, not created (`docs/HOSTING.md`)
- [ ] Public URL verified

## Documentation & submission
- [ ] README (quickstart, results, limitations)
- [ ] `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/RESULTS.md`, API contract
- [ ] Clean git history
- [ ] Final submission checklist reflects reality
