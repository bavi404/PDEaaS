# ZeroGPU Gradio Space (M7, implemented and validated locally; **not deployed**)

This is the zero-cost hosting route chosen in [`FREE_HOSTING_FEASIBILITY.md`](FREE_HOSTING_FEASIBILITY.md): a Hugging Face
**Gradio SDK** Space on **ZeroGPU** hardware. Free personal accounts in good standing (verified email, account older
than 30 days, as confirmed by the owner) may host up to 2 such Spaces.

**Unchanged:** the model, checkpoint (SHA-256 `d7133c7c…124df7`), preprocessing, inference code, FastAPI app, Next.js
source and the Docker deployment.

## 1. Architecture

```
Gradio SDK Space · python 3.12.12 · app.py → uvicorn :7860 (one process)
 ├─ /api/*     pdeflow.api.app.create_app(): unchanged; its lifespan runs (SHA check, model + samples load)
 ├─ /gradio    gr.Blocks UI + Gradio API (/predict_benchmark, /predict_custom): same Predictor, same sample store,
 │             same validate_permeability as /api/predict
 └─ /          existing Next.js frontend, static export; the browser calls same-origin /api/*
```

| Decision | Alternatives | Justification |
|---|---|---|
| FastAPI root, existing API mounted at `/api`, Gradio via `gr.mount_gradio_app` at `/gradio` | Gradio-only app | Keeps the validated REST API and the full web frontend. Gradio is the SDK requirement and gives a second UI and API. |
| **CPU inference, no `@spaces.GPU`** | GPU inside `@spaces.GPU` | 12–30 ms per request on CPU. No GPU queue, and no visitor GPU quota consumed (2 min/day unauthenticated). CPU numerics are bit-identical to the evaluated results; cuFFT would not be. |
| `import spaces` first (optional import) | not importing it | ZeroGPU documents the import. It must precede torch. It is effect-free elsewhere and absent in the Python 3.11 dev env. |
| **torch 2.13.0 CUDA build** (`+cu130`) | CPU wheel (feasibility plan) | **Measured:** under emulated ZeroGPU (`SPACES_ZERO_GPU=true`), `import spaces` crashes with the CPU-only wheel (`torch.version.cuda` is `None`). The CUDA build works and runs our CPU inference bit-identically. 2.13.0 is the newest version on ZeroGPU's documented list. |
| Python **3.12.12** | 3.11 (dev, Docker) | ZeroGPU offers only 3.12.12 or 3.10.13. The 3.11 dev environment and Docker image are unchanged; both pass the same tests. |
| `pillow` 12.3.0 in the Space | 13.0.0 (dev lock) | gradio 6.30.0 requires `<13`. Pillow is not on the inference path. Every other shared package keeps its dev-lock version (constraints in `uv pip compile`). |
| Frontend static export built from a temp copy (`build_frontend.py`): `src/app/api` removed, `output: "export"` | change `frontend/` | The source and the Docker build stay untouched. The proxy route isn't needed on the same origin. `build-info.json` records commit and tree hash. |
| Clean-URL static route accepting GET **and HEAD**, path-traversal safe | `StaticFiles(html=True)` | The existing e2e suite caught both requirements in the prototype: `/results` (not `/results/`), and HEAD prefetches from the Next.js client. |
| Gradio predictions also recorded in `/api/activity` and logged as `prediction` events (`channel: "gradio"`) | not recorded | The dashboard "recent activity" reflects all predictions served by the process. |

## 2. Files

| File | Purpose |
|---|---|
| `deploy/zerogpu/app.py` | Space entrypoint: `build_app()`, Gradio UI, `predict_benchmark` / `predict_custom` (API-identical code paths), static route |
| `deploy/zerogpu/build_frontend.py` | static export of the existing frontend → `deploy/zerogpu/web/` (gitignored) |
| `deploy/zerogpu/requirements.in`, `requirements.txt` | Python 3.12 lock for the Space (114 packages; generated with constraints from `backend/requirements.lock`) |
| `deploy/zerogpu/requirements-dev.txt` | + pytest for local tests |
| `deploy/zerogpu/README.md` | Space card: `sdk: gradio`, `sdk_version: 6.30.0`, `python_version: 3.12.12`, `app_file: app.py`, `license: mit`, attribution |
| `deploy/zerogpu/tests/` | 31 tests for the integrated app |
| `scripts/deploy_zerogpu_space.py` | stages the Space repo from `git archive HEAD` + export (dry-run; upload only with approval) |
| `frontend/e2e/gradio.spec.ts` | Playwright for the Gradio UI (`E2E_GRADIO=1`) |
| `scripts/verify_api_inference.py` | `--out` option; the default output and committed M4 report are unchanged |

## 3. Run locally

```bash
uv venv --python 3.12.12 .venv-zerogpu
```

```bash
uv pip install --python .venv-zerogpu/bin/python --no-deps -r deploy/zerogpu/requirements-dev.txt
```

```bash
.venv/bin/python deploy/zerogpu/build_frontend.py            # -> deploy/zerogpu/web
```

```bash
PORT=7860 .venv-zerogpu/bin/python deploy/zerogpu/app.py      # http://localhost:7860  /gradio  /api/docs
```

**Defaults** (environment variables still win):
- `PDEFLOW_NUM_THREADS = min(4, cpu_count)`
- `PDEFLOW_CORS_ORIGINS = ""`
- `PDEFLOW_WEB_DIR` overrides the export location

Without an export, `/` redirects to `/gradio/`.

## 4. Validation (all local; evidence in `reports/m7_zerogpu/`)

| Gate | Result |
|---|---|
| Backend strict suite, Python 3.11 (dev) | **151 passed**, 0 skipped |
| Backend strict suite, Python 3.12.12 (Space lock, torch 2.13.0+cu130) | **151 passed**, 0 skipped |
| Space app tests (3.12) | **31 passed**:
  - routes and checkpoint identity; wrong SHA refused
  - `/api` responses identical to a standalone `create_app()`
  - static pages GET+HEAD, assets, 404, path traversal
  - attribution and licence
  - **Gradio ↔ FastAPI exact equality** (prediction SHA-256 and metrics) for 4 benchmark and 4 custom cases, plus over HTTP via `gradio_client`
  - validation parity; activity recording
  - emulated ZeroGPU subprocess bit-identical |
| 1,000-sample inference equivalence vs `m2_eval_fno_baseline.json` | **bit-identical** (max rel diff 0) in 3 environments: Python 3.11/torch 2.15.0 (committed M4 report byte-unchanged), 3.12.12/torch 2.13.0+cu130, and the same under emulated ZeroGPU (`spaces` patched torch) |
| Frontend typecheck + Vitest | clean, **52 passed** |
| Production builds | standalone (Docker path) ✓; static export ✓ (6 pages, 5.4 MB) |
| Playwright: existing suite against the **integrated app** | **24/24 passed**. The 2 not run are the dead-backend tests: no separate backend exists in one process. The in-browser "backend failure during prediction" tests are included and pass. |
| Playwright: Gradio UI spec | **6/6 passed** (metric equals `/api/predict`, attribution, custom with no ground truth, OOD, validation error) |
| Playwright: existing suite against the original stack (regression) | **26/26 passed** |
| `verify_deployment.py` against the integrated app | all checks pass: SHA, model card, samples, predictions max \|Δ\| **0.0** vs a separate in-process API |
| Checkpoint SHA-256 | `d7133c7c681cf3e9632579833fe357636d84ff7a5678d94b2f24c38955124df7`, unchanged (checked by tests and by the stager) |
| **Staged Space repository**: `deploy_zerogpu_space.py --dry-run --stage-to` (73 files, 11.1 MB, from commit `444de20`), app run from the staged directory | healthy in **4.7 s**. Loads only the staged copy. `verify_deployment.py` all pass; Playwright existing suite **24/24**, Gradio **6/6**. RSS after all tests **820 MiB**. |

## 5. Remaining ZeroGPU hosting-policy and platform risks (unverifiable before deployment)

1. **CPU-only use of a ZeroGPU Space is not documented as intended use.** The `spaces` code tolerates it (no decorated function → no GPU registration), but Hugging Face could change behaviour or policy.
2. **The ZeroGPU host's CPU/RAM are not documented.** The staged app measured **820 MiB RSS** after all tests, with the CUDA build of torch; the CPU-wheel prototype was 501 MiB, and RSS includes mapped CUDA libraries. Latency will be measured after deployment.
3. **Serving a FastAPI root via uvicorn instead of `demo.launch()`** works locally and under emulation, but has not run on HF. If the platform requires `launch()`, the fallback is to mount the API and static routes on Gradio's own app.
4. **ZeroGPU environment specifics:** on ZeroGPU, Gradio's queue defaults `max_size=1` for Gradio events, which affects only `/gradio`. `/api` is served directly by FastAPI. Torch is patched for CUDA emulation; inference is pinned to CPU (`load_checkpoint`/`predict` use `cpu`), as verified by the emulation test.
5. **Install size:** the CUDA build of torch makes the Space's pip install about 5 GB. That's normal for ZeroGPU Spaces, but the first build is slow.
6. **Sleep:** idle free Spaces sleep, and the first visit cold-starts (app startup about 1–2 s locally after imports).
7. **Single process:** if the API fails, the whole Space is down. The platform restarts it.

## 6. Deployment (NOT done; requires the owner's separate approval)

**Authentication: host-scoped proxy secret.** The sandbox proxy adds the Hugging Face credential to requests for
`huggingface.co`, and the token never enters the shell.
- **Identity:** `scripts/deploy_zerogpu_space.py` checks it with a **token-less** `GET /api/whoami-v2` and prints only the account name, type and token role.
- **Hub calls:** every Hub call uses `HfApi(token=False)`, so no local token or `HF_TOKEN` is ever read or sent; the proxy supplies the credential.
- **Uploads:** they fetch a short-lived Xet/LFS token from `huggingface.co` (also proxy-authenticated) and use it on the storage hosts. `cas-server.xethub.hf.co` was verified reachable.
- **Tested:** against a local mock Hub, which shows that no `Authorization` header is ever sent, under `huggingface_hub` 2.2.0 and 2.4.0.

**The script never creates a Space.** The owner creates `bavi404/pdeflow` on the website: public, Gradio SDK, ZeroGPU hardware. Before any upload, a **read-only preflight** checks:
- the Space exists
- its owner equals the authenticated account and the requested owner
- `sdk == "gradio"`
- it is public
- its hardware is ZeroGPU (`zero-*`)

Uploading additionally requires `--confirm-deploy <owner/space>`.

```bash
# stage only (no network)
.venv/bin/python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --dry-run
```

```bash
# read-only: proxy authentication + preflight of the existing Space; uploads nothing
uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --preflight-only
```

```bash
# deploy: only after explicit owner approval
uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 python scripts/deploy_zerogpu_space.py --space bavi404/pdeflow --confirm-deploy bavi404/pdeflow --wait 2400
```

```bash
# public verification
.venv/bin/python scripts/verify_deployment.py --url https://bavi404-pdeflow.hf.space
```

```bash
cd frontend && E2E_BASE_URL=https://bavi404-pdeflow.hf.space npx playwright test e2e/app.spec.ts --trace off --grep-invert "dashboard, workspace and model page show error states"
```

```bash
cd frontend && E2E_GRADIO=1 E2E_BASE_URL=https://bavi404-pdeflow.hf.space npx playwright test e2e/gradio.spec.ts --trace off
```
