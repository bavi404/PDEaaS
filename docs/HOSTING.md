# Hosting status and limitations

## 1. Live deployment: Render Free

| | |
|---|---|
| **Public URL** | **https://pdeaas.onrender.com** (web app). Gradio is not part of this deployment; the Docker image serves the Next.js app plus FastAPI. |
| Platform | Render **Free** web service, created by the owner. Built from the unchanged repository `Dockerfile`; configuration in §4. |
| API | `https://pdeaas.onrender.com/api/health`, `/api/model`, `/api/samples`, `/api/samples/{id}`, `/api/predict`, `/api/activity` |
| Status | **verified working** twice: 2026-11-12 **06:02 UTC** and **10:48 UTC** (evidence in `reports/m7_render/`) |

### Verification results
Both runs used `scripts/verify_deployment.py` from this sandbox; results were identical in both.

| Check | Result |
|---|---|
| `/api/health` | 200, `model_loaded: true`, **checkpoint SHA-256 `d7133c7c…124df7` = the evaluated checkpoint**, 24 samples |
| `/api/model` | identical to the local API (except `num_threads`, which is 1 on Render) |
| `/api/samples` | identical to the local API (24 held-out PDEBench test samples) |
| Real predictions: test samples 4701 (128² direct and default), 2113 (64²), 8142 (128²), plus a 128² custom field | all **match the local validated pipeline** within max \|Δu\| = **4.2 × 10⁻⁷** (float32 rounding on a different CPU; tolerance 1 × 10⁻⁵) |
| Report worst case (test 4701, 128² direct) | rel-L2 vs official target **0.2440539** vs report 0.2440540 (relative difference 1.7 × 10⁻⁷) |
| Validation error (malformed field) | 422 with request ID |
| Warm latency, 10 sequential requests through the public URL | 64²: median 339–371 ms. 128² default: median 323–432 ms. Server-side inference 40–66 ms. |
| Playwright browser suite against the public URL | **26/26 passed** at both times |

The one suite test not run needs a second web server pointed at a closed API port, which cannot exist on Render. The
in-browser "backend failure during prediction" test is included and passes.

Playwright's Node request client needs the sandbox's HTTPS proxy. A temporary, untracked config passed
`use.proxy` (with `NODE_EXTRA_CA_CERTS`). Without it, 12 API-level tests failed with `getaddrinfo ENOTFOUND`, a
harness networking issue and not an application failure.

## 2. Sleeping and cold starts (observed, not assumed)

Render Free **spins a service down after 15 minutes without traffic** and wakes it on the next request. Render
documents the wake-up as "about one minute". The app then needs **24 s at 0.5 CPU to 172 s at 0.1 CPU** to import
PyTorch, verify the checkpoint and load the model (measured locally under `--memory=512m`).

**Observed on 2026-11-12:**
1. **06:06–06:23 UTC:** no requests from this session; the service went to sleep.
2. **06:23:42 UTC:** wake-up request. **No HTTP response** at all; the connection dropped after 300 s.
   - **06:30–06:54 UTC:** 21 further requests, one a minute, all timed out.
   - In every case TLS to Render's edge succeeded (and render.com itself responded), but the service did not answer.
3. **By 10:48 UTC** the owner reported the Render logs showing the service live again. It answered in 0.38–0.63 s and passed full verification (§1).

So the first wake-up after idling **did not complete within at least 30 minutes**. The cause is not determined from
this side (it could be a slow wake-up, a failed restart, or a platform problem); the owner's Render logs are the source
for it. **Visitors should expect the first request after an idle period to be slow or to fail**, and should retry after
a few minutes. Once awake, the service is responsive. Raw probe output: `reports/m7_render/sleep_wake_observation.txt`.

## 3. Free-tier limitations

- **Sleep and cold starts:**
  - The service spins down after 15 min idle.
  - Waking takes at least ~1 min (Render) plus 24–172 s (app start). One observed wake-up did not complete within 30 min (§2).
  - The first visitor after an idle period pays this cost.
- **Resources:** **512 MB RAM** and **"less than 1 CPU"** (the exact share is not documented).
  - The app needs ~290 MiB of non-reclaimable memory.
  - Measured per-process peaks: 316 MiB (API) and 98 MiB (web).
  - No OOM under a 512 MB limit locally (§4).
- **CPU inference:**
  - There is no GPU; the FNO runs on CPU by design, which keeps results identical to the validated benchmark.
  - Predictions take a few hundred ms per request (server inference 40–66 ms on Render).
  - One model forward runs at a time, so concurrent visitors queue.
- **Usage caps:** 750 free instance-hours per month. Outbound bandwidth and build minutes are capped monthly; without a payment method Render **suspends** the service rather than billing.
- **Ephemeral state:** no persistent disk. The dashboard's "recent prediction activity" is in memory and **resets on every sleep or redeploy**.
- **Terms can change:** free-tier facts are from Render's documentation as read on 2026-11-11.

## 4. Configuration and local emulation

**Render settings:**
- Docker runtime, `./Dockerfile`, build context `.`
- Free instance type
- Health check path **`/api/health`**
- Environment: `PORT=7860`, `HOSTNAME=0.0.0.0`, `PDEFLOW_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; no secrets.

**Before creation:** the unchanged image was emulated locally with `--memory=512m --memory-swap=512m` at 0.5 and 0.1 CPU:
- healthy after 24 s and 172 s
- the entrypoint used 21 of its 120 readiness polls at 0.1 CPU
- the health check answered in < 0.6 s during 4 concurrent predictions
- no OOM

## 5. Earlier attempts (Hugging Face)

| Option | Outcome |
|---|---|
| Hugging Face **Docker** Space | needs a **paid PRO plan** to create; not attempted ([`DEPLOYMENT.md`](DEPLOYMENT.md)) |
| Hugging Face **ZeroGPU Gradio** Space ([`ZEROGPU.md`](ZEROGPU.md)) | two controlled attempts (2026-11-11), **both failed** |

- **Attempt 1:** build failure. HF installs `gradio[oauth,mcp]`, which needs `pydantic<=2.12.5`. Fixed in `0172fb2`, and `deploy/zerogpu/check_hf_build.py` now reproduces HF's install command first.
- **Attempt 2:** built, then the runtime rejected the app with `No @spaces.GPU function detected during startup`, and port 7860 was already in use. ZeroGPU requires GPU-decorated functions, while this app runs inference on CPU by design.
  - Making it work would mean GPU inference: it uses visitor quota, and its numerics cannot be validated in this GPU-less environment.
  - Adding an unused GPU function only to pass the check was rejected as circumventing the platform's intent.

The Space `bavi404/pdeflow` remains in its failed state; it is not the deployment.
