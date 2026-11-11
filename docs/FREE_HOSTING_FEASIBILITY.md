# Zero-cost public hosting: feasibility report (M7)

**Status: investigation only.** Nothing was created, deployed or changed in the application. All prototypes ran in a
scratch directory. The evidence is in `reports/m7_feasibility/`. Docs were read on 2026-11-11 through the sandbox proxy;
copies are in `reports/m7_feasibility/docs_snapshot/`.

## 0. Correction to M6

The M6 plan (Docker Space on "free CPU basic") is **not zero-cost**. The current Hugging Face docs (`spaces-overview.md`, `spaces-gpus.md`) say:

> "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for
> personal accounts, Team or Enterprise for organizations. Free personal accounts in good standing can still host up to
> 2 Gradio Spaces running on ZeroGPU."

`docs/DEPLOYMENT.md` assumed that CPU basic was free to *create*. It has no hourly cost, but creating a Docker Space needs PRO. The Docker image itself remains valid, and it is the natural artefact for Render or any paid host.

## 1. Free-tier eligibility (exact, from official docs)

| | Hugging Face ZeroGPU (Gradio SDK) | Render Free web service |
|---|---|---|
| Who | "Free personal accounts: accounts in good standing (**verified email, account older than 30 days**) can host up to **2** ZeroGPU Spaces for free." | Any workspace. Free web services are offered without a payment method; without one, overages **suspend** services instead of billing. |
| SDK / runtime | "ZeroGPU Spaces are **exclusively compatible with the Gradio SDK**." Python **3.12.12 or 3.10.13**. PyTorch "almost all versions from 2.8.0 to latest" (listed: 2.8.0 … 2.13.0). Gradio 4+. | Docker or native Python. Custom Docker containers are listed. |
| Compute | GPU only inside `@spaces.GPU` functions. The CPU/RAM of the host is **not documented** (the general Spaces limit is 16 GB RAM / 2 vCPU). | **512 MB RAM, "less than 1 CPU"** |
| Visitor limits | GPU quota per visitor per day: unauthenticated **2 min**, free account 5 min. Applies only to GPU time. | 750 free instance-hours per month per workspace |
| Idle behaviour | Free-hardware Spaces sleep after inactivity ("currently, 48 hours" for cpu-basic; the docs don't say whether ZeroGPU is the same) | **Spins down after 15 min idle; spin-up "takes about one minute"**; ephemeral disk |
| Owner action needed | Confirm the `bavi404` account has a verified email and is **older than 30 days** | Create a Render account; connect the GitHub repo |

## 2. Runtime and dependency compatibility (measured)

| Check | Result |
|---|---|
| Our stack under **Python 3.12.12** (ZeroGPU's version) | resolves; the only change is `pillow` 13.0.0 → 12.3.0 (gradio requires `<13`; Pillow is not on the inference path). fastapi 0.162.4, starlette 1.10.0, pydantic 2.14.3, numpy 2.4.6, uvicorn 0.54.0 unchanged; plus gradio 6.30.0 and spaces 0.51.3 |
| Full 1,000-sample inference-equivalence check vs `reports/m2_eval_fno_baseline.json`, Python 3.12.12 + torch **2.15.0** | **bit-identical** for all three strategies (max rel diff 0) |
| Same, with torch **2.13.0** (latest on ZeroGPU's documented list) | **bit-identical** (max rel diff 0) |
| Strict backend suite (151 tests) under Python 3.12.12 | **151 passed** |
| CPU inference without GPU allocation | Yes, technically: the docs say `@spaces.GPU` is "effect-free in non-ZeroGPU environments", and the GPU is only requested inside decorated functions. In `spaces` 0.51.3 the startup returns without registering anything if no function is decorated (`spaces/zero/__init__.py`). Not calling `@spaces.GPU` means **no GPU request and no visitor quota**. The docs do **not** state that CPU-only ZeroGPU Spaces are an intended use. |
| Python 3.11 requirement (M6) | ZeroGPU does not offer 3.11. That deployment must use **3.12.12**, a justified, measured deviation; the Docker image stays 3.11. |

**Render emulation:** the unchanged M6 image under `docker run --memory=512m --memory-swap=512m`; details in `reports/m7_feasibility/render_*`.

| Limit | Startup to healthy | Prediction (128², warm) | Memory | OOM |
|---|---|---|---|---|
| 0.5 CPU | **58 s** | 0.11–0.28 s (first 4.1 s) | anon RSS **277 MiB**; peak usage 308 MiB, up to 512 MiB including reclaimable page cache | no |
| 0.1 CPU | **174 s** | **0.9–1.3 s** | cgroup peak 322 MiB | no |

So the image fits in 512 MB, with about 200 MiB of anonymous-memory headroom, but:
- Render's documented CPU is "less than 1" and its exact share isn't stated.
- A cold start = spin-up (~1 min) + **1–3 min** of app startup.

## 3. Proposed architecture (ZeroGPU, CPU-only inference)

```
Gradio SDK Space (python_version 3.12.12, ZeroGPU hardware), app.py → uvicorn :7860, one process
 ├─ /api/*     existing pdeflow.api.app.create_app(), unchanged (pinned-SHA check, Predictor, JSON logs)
 ├─ /gradio    small gr.Blocks UI calling the SAME Predictor (also gives the Gradio API)
 └─ /          existing Next.js frontend as a static export (built at deploy time; same-origin /api)
```

**Verified with a throwaway prototype** (`reports/m7_feasibility/prototype/`, run locally, not on ZeroGPU):
- `/api/health` returns the pinned SHA.
- Sample 4701 matches the report; Gradio `/solve` and `/api/predict` agree.
- The **existing Playwright suite passes 22/22** (all except the dead-backend tests, which need a second server), plus 2/2 of the in-browser failure tests.
- **Process RSS** is 501 MiB, measured before the static frontend was mounted.

The suite caught two real defects in the prototype before it passed:
- the URL shape (`/results/` vs `/results`)
- `HEAD` prefetches returning 405

Both are fixed in the plan below.

**Why CPU-only rather than `@spaces.GPU`:**
- **Speed:** CPU inference takes 12–16 ms per request. A GPU call adds allocation and queueing per request.
- **Quota:** a GPU call consumes visitors' 2–5 min/day of GPU quota.
- **Numerics:** CPU keeps the evaluated numerics bit-identical; cuFFT would not.

## 4. Required code changes (smallest set; nothing replaces the validated pipeline)

1. `deploy/zerogpu/app.py` (new, about 60 lines):
   - root FastAPI with `lifespan` delegating to `create_app()`'s lifespan, and `create_app()` mounted at `/api`
   - Gradio Blocks at `/gradio` (sample picker, resolution, strategy → Plotly figure + metrics, official target and reference kept separate)
   - clean-URL static route (GET + HEAD) for the frontend export
   - `uvicorn.run(..., port=7860)`
2. `deploy/zerogpu/requirements.txt` (new): the Python 3.12 lock (`torch==2.13.0+cpu` CPU index, or 2.15.0), `gradio==6.30.0`, `spaces==0.51.3`, `pillow==12.3.0`. Plus a Space card with `sdk: gradio`, `sdk_version: 6.30.0`, `python_version: 3.12.12`, `app_file: app.py`, `license: mit`.
3. **Frontend static-export build** (deploy-time only): copy `frontend/`, drop `src/app/api` (the proxy isn't needed on the same origin), `output: "export"`. **No change** to the existing app source.
4. `scripts/deploy_zerogpu_space.py`: stages the files above from `git archive HEAD` plus the built export; reuses the token and visibility guards of `deploy_hf_space.py`.
5. **Tests:**
   - an API test that `create_app()` works when mounted under `/api` with a delegated lifespan
   - the equivalence check run in the 3.12 environment
   - Playwright against the staged app

   Existing tests are unchanged.

The Docker/Next.js server deployment, the API, the model, the checkpoint and all tests stay as they are.

## 5. Risks and limitations

| Risk | Severity | Mitigation |
|---|---|---|
| Account eligibility: email not verified or account < 30 days | blocking | owner confirms before creation |
| A ZeroGPU Space that never requests the GPU is not documented as intended use; HF could change behaviour or policy | medium | the `spaces` code allows it today; fallback below |
| ZeroGPU host CPU/RAM not documented | medium | measure after deploy (startup time, latency); the app needs ~0.5 GB |
| Uses `gr.mount_gradio_app` + uvicorn instead of `demo.launch()` | low–medium | prototype works locally; if the platform requires `launch()`, serve via Gradio's own app and add the API routes to it |
| Static export drops the Next.js proxy and the "dead backend" path (one process: if the API is down, so is the site) | low | the in-browser failure test still passes; the health page covers the API |
| Python 3.12 / torch 2.13 differ from the M2 environment | low | measured bit-identical on all 1,000 test samples |
| Free Spaces sleep when idle (first visit cold-starts; about 9 s app startup measured on 2 vCPU in Docker) | low | documented |
| **Render:** fits 512 MB, but the CPU share is undocumented; cold start 1 min + 1–3 min app start; 15-min idle spin-down | high for UX | not recommended as the primary host; the image works unchanged as a fallback |

**Fallback if ZeroGPU is refused or fails:** Render Free with the unchanged M6 Docker image. It fits in memory as measured, with slow cold starts. Alternatively, a Static Space with in-browser inference is free for everyone, but it would need an ONNX/WebAssembly port of the FNO. That would replace the validated pipeline and is not recommended.

## 6. Deployment and verification plan (after approval)

1. **Owner:** confirm the account is older than 30 days with a verified email; set `HF_TOKEN` (write).
2. **Implement §4 changes; local gates:**
   - strict backend suite under 3.11 **and** 3.12
   - equivalence check in the 3.12 environment, bit-identical
   - Playwright 22/22 + failure tests against the staged app run locally
3. **Deploy** to `bavi404/pdeflow` as a **Gradio** Space (ZeroGPU hardware, public, MIT). Same guards as before: token from env only, clean tree, visibility check, `--wait`.
4. **Public verification:**
   - `scripts/verify_deployment.py --url https://bavi404-pdeflow.hf.space`: health/SHA, model, samples, predictions element-wise vs local, latency
   - Playwright against the public URL
   - record startup time, cold start, latency and platform behaviour in `docs/DEPLOYMENT.md`
5. **If ZeroGPU rejects the Space or CPU-only use:** report it, then propose Render Free (Docker image as is) with the same verification.
