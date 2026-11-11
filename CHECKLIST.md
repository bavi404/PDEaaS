# PDEFlow Task Checklist

Legend: `[x]` done and verified · `[ ]` open · `[!]` blocked (reason given)

## M0 — Discovery & planning
- [x] Inspect working dir, git, Python, packages, disk, memory, GPU/CUDA, Node, Docker (`docs/ENVIRONMENT.md`)
- [x] Architecture, decisions, risks, milestones (`docs/PLAN.md`)
- [x] Repository skeleton + git init
- [!] Push to GitHub — GH token invalid in this environment

## M1 — Data
- [!] Install Python deps (torch CPU, h5py, fastapi, pytest) — PyPI unreachable
- [!] Query DaRUS for Darcy file sizes — host denied by network policy
- [ ] Download one β file (or range-read N samples), record SHA-256 + URL
- [ ] Inspect HDF5 keys/shapes/dtypes → `docs/DATA.md`
- [ ] Seeded split manifest (train/val/test), train-only normalization stats
- [ ] Data unit tests (shapes, no index overlap between splits, determinism)

## M2 — Model & training
- [ ] FNO via `build_model(cfg)` (neuraloperator or in-house, decision recorded)
- [ ] Model unit tests (shape, resolution invariance, param count)
- [ ] Training loop: seeds, device=auto, AMP on CUDA, grad clipping, LR schedule
- [ ] Checkpoint save/load (weights + config + normalizer + split hash)
- [ ] Train/val loss log (JSON/CSV) + plot
- [ ] Real training run completed

## M3 — Evaluation
- [ ] Test relative L2 (mean, median, p90, worst-case)
- [ ] 128×128 zero-shot evaluation
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
- [!] Docker base image pulls — Docker Hub unreachable
- [ ] Dockerfiles (api, web) + docker-compose.yml with health checks
- [ ] Compose verified end-to-end
- [ ] Public deployment live and verified

## M7 — Documentation & submission
- [ ] README (quickstart, results, limitations)
- [ ] `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/RESULTS.md`, API contract
- [ ] Clean git history
- [ ] Final submission checklist reflects reality
