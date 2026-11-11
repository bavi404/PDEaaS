# PDEFlow — Architecture, Decisions, Risks, Milestones

Session start: **2026-11-11 ~07:23 UTC** → hard deadline **2026-11-14 ~07:23 UTC**.

## 1. Problem statement

Learn the solution operator `G: a(x) ↦ u(x)` of steady-state 2D Darcy flow on the unit square

    -∇·(a(x) ∇u(x)) = β,   x ∈ (0,1)²,   u = 0 on ∂(0,1)²

where `a` is a piecewise-constant permeability field and `β` a constant forcing term,
using a Fourier Neural Operator trained on the PDEBench Darcy Flow data, and serve it as an API with an
interactive web UI.

## 2. Architecture

```
┌──────────────────────────┐   HTTPS/JSON   ┌───────────────────────────────────┐
│ Frontend (Next.js + TS)  │ ─────────────▶ │ Backend (FastAPI, Python)          │
│ - sample picker          │                │  /health  /model  /samples         │
│ - coefficient editor     │ ◀───────────── │  /samples/{id}  /predict           │
│ - Plotly heatmaps: a, û, │                │  pydantic validation, JSON logs    │
│   u_true, |û-u|, metrics │                │  ModelService (loaded once)        │
└──────────────────────────┘                └──────────────┬────────────────────┘
                                                           │ torch (CPU; CUDA optional)
                         offline pipeline                  ▼
 PDEBench HDF5 ─▶ scripts/download ─▶ preprocess ─▶ train ─▶ checkpoints/*.pt ─▶ evaluate ─▶ reports/*.json
 (DaRUS, β=1.0)   (size check, sha256)  (subset, split,       (seeded, logged)                (rel-L2, latency)
                                         train-only stats)
```

Repository layout (Python package `pdeflow` is shared by the offline pipeline and the API, so preprocessing
and normalization are identical at train and serve time):

```
backend/pdeflow/{data,models,training,evaluation,api}   backend/tests   backend/configs
frontend/   scripts/   deploy/   docs/   data/{raw,processed} (gitignored)   checkpoints/ (gitignored)
```

## 3. Key decisions (alternatives → choice → justification)

| # | Decision | Alternatives | Choice | Why |
|---|---|---|---|---|
| D1 | Data subset | all 5 β files (~several GB); β-conditioned model; one β | **One β file (β = 1.0), first N≈2000 samples** | One file fits disk/RAM/CPU budget; operator is well defined for fixed β; β=1.0 is the setting most often reported for PDEBench Darcy. File size **will be checked via the DaRUS API before download**. |
| D2 | Training resolution | native 128×128; 64×64 (stride-2 subsample) | **64×64 train, also evaluate at 128×128** | ~4× cheaper on CPU; FNO is discretisation-invariant, so 128×128 evaluation is a meaningful zero-shot super-resolution test (reported honestly, whatever it shows). |
| D3 | Splits | random per run; fixed manifest | **Seeded permutation saved to `splits.json` (+ source file SHA-256)**, 80/10/10 | Reproducible; split fixed before any statistics are computed. |
| D4 | Normalization | global; per-sample; train-only Gaussian | **Pointwise/global stats from train split only** | Prevents test leakage; same stats saved in checkpoint and used by API. |
| D5 | Model implementation | `neuraloperator` FNO; in-house FNO2d | **`neuraloperator` if it installs cleanly on our torch/Python; in-house FNO2d (~100 LOC, unit-tested) otherwise**, behind one `build_model(cfg)` factory | Respects the stated framework preference while keeping a working path if the library's pinned deps conflict with Python 3.13 / current torch. Decided after an install test, recorded in docs. |
| D6 | Loss / metric | MSE; relative L2 | **Relative L2 for training and evaluation** | Standard metric in the FNO and PDEBench papers; scale-invariant across samples. |
| D7 | Compute | GPU only; CPU only | **CPU-first, `device=auto`; AMP only on CUDA** | No GPU in this container. Small FNO (width 32, 12 modes, 4 layers) at 64×64 is trainable on 4 vCPUs in hours. Note: AMP + FFT needs care (cuFFT half precision requires power-of-2 sizes; spectral weights kept in fp32). |
| D8 | Serving runtime | ONNX Runtime (pre-installed); TorchScript; plain torch | **Plain torch, CPU** | ONNX export of `torch.fft.rfft2/irfft2` is poorly supported; torch CPU latency for a ~1–2M-param FNO at 64² is expected to be in the ms range (to be measured, not assumed). |
| D9 | API style | gRPC; REST/JSON | **REST/JSON via FastAPI + pydantic** | Simple browser client; OpenAPI contract generated automatically. 64×64 arrays ≈ 4k floats → small payloads. |
| D10 | Frontend | Streamlit; Next.js | **Next.js (App Router) + TS + react-plotly.js** | Per scope. Plotly loaded via dynamic import with `ssr:false` (it touches `window`). Tests: Vitest + React Testing Library. |
| D11 | Packaging | separate images; single image | **Compose with two services (`api`, `web`)** locally; public deploy target chosen in M6 | Clear separation; health-checked startup ordering. |
| D12 | Public deployment | Hugging Face Spaces (Docker); Render; Fly.io | **Recommended: HF Spaces (Docker SDK)** — free CPU tier has enough RAM for torch. Render free (512 MB) is tight for torch. | Needs the user's account + token, and the host allowlisted. Deployment status will be reported exactly as observed. |

## 4. Risks and dependencies

| Risk | Impact | Mitigation / status |
|---|---|---|
| **Network policy blocks PyPI, npm, pytorch.org, DaRUS, Docker Hub** | Blocks everything past planning | **BLOCKER** — user must widen network access (see §6). |
| **GitHub token invalid** | Can't push / trigger deploys | User reconnects GitHub; until then commits stay local. |
| No GPU in container | Slower training | CPU-sized config; optional remote run on the user's RTX 5070 via Remote Control. |
| `neuraloperator` vs Python 3.13 / torch compatibility | Install failure | D5 fallback; library choice recorded. |
| PDEBench file larger than expected / slow mirror | Disk/time | Check `Content-Length` first; consider HTTP range reads (h5py + fsspec) to fetch only N samples. |
| HDF5 key layout differs from docs | Loader bugs | Inspect keys/shapes/dtypes and log them in `docs/DATA.md` before writing the loader. |
| Docker image pulls blocked | No container build / compose test | Needs Docker Hub egress; otherwise compose files are written but **reported as unverified**. |
| Free-tier cold starts / RAM | Flaky public demo | Small model, lazy load once at startup, health check. |
| 72 h budget | Scope creep | Milestones below; stretch goals clearly marked. |

## 5. Milestones (hours from session start)

| M | Window | Deliverable | Exit criterion |
|---|---|---|---|
| M0 | 0–2 h | Discovery, plan, repo skeleton, checklist | This document committed |
| M1 | 2–8 h | Deps; dataset inspection, subset download, provenance, splits, normalization | `docs/DATA.md` with real sizes/shapes/SHA-256; data tests pass |
| M2 | 8–20 h | FNO model, training loop (seeds, checkpoints, loss curves), real training run | Checkpoint + `train_log.json` from an actual run |
| M3 | 20–28 h | Evaluation: test rel-L2 stats, 128² super-res eval, latency benchmark | `reports/*.json` generated by scripts; `docs/RESULTS.md` quoting them |
| M4 | 28–40 h | FastAPI service, validation, logging, pytest suite | `pytest` green; OpenAPI doc |
| M5 | 40–54 h | Next.js UI + tests, wired to API | `npm test` + build green; manual E2E check |
| M6 | 54–64 h | Dockerfiles, Compose, public deployment | `docker compose up` verified; public URL responds |
| M7 | 64–72 h | README, architecture doc, submission checklist, buffer | Checklist accurate |

## 6. What is needed from the user right now

1. **Network access**: allow package managers (pypi.org, files.pythonhosted.org, registry.npmjs.org) and add
   `darus.uni-stuttgart.de`, `download.pytorch.org`, and for Docker builds `registry-1.docker.io`,
   `auth.docker.io`, `production.cloudflare.docker.com`; for deployment `huggingface.co` (if HF Spaces).
2. **GitHub**: reconnect so pushes work; name the target repository.
3. **Deployment**: confirm target (HF Spaces recommended) and provide its token as an environment secret.
