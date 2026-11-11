# Deployment: Hugging Face Spaces (Docker SDK)

**Status: not deployed.** This document is the plan and the required configuration. Deployment will be done in M6, only
after authorization.

## 1. Space configuration

The Space repository needs a `README.md` with YAML front matter:

```yaml
---
title: PDEFlow
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860        # HF routes public traffic to this port
pinned: false
license: mit          # code licence (TBD by the owner); the bundled PDEBench samples are CC BY 4.0 (attributed)
---
```

| Setting | Value | Reason |
|---|---|---|
| Hardware | **CPU basic** (free: 2 vCPU, 16 GB RAM) | Measured needs: about 1.1 GB peak RSS in training, much less for inference. The model is 357k parameters, and forward time on 4 vCPU is 10–18 ms per sample. |
| SDK | Docker | Full control of the Python 3.11 + torch CPU environment and of the Next.js build (M5) |
| Port | 7860 | HF Docker default (`app_port`) |
| Container user | non-root UID 1000 | HF requirement for Docker Spaces |
| `PDEFLOW_NUM_THREADS` | 2 | matches CPU basic vCPUs; avoids oversubscription |
| `PDEFLOW_CORS_ORIGINS` | empty if the frontend is served from the same origin | single-container deployment, no cross-origin calls |
| Visibility | public | required by the brief |

**Layout:** one container serves the Next.js frontend and proxies `/api/*` to uvicorn on localhost. The details are settled in M6.

**Payload:** the image contains the committed checkpoint (2.8 MB, SHA pinned), the demo samples (2.8 MB) and `reports/m2_eval_fno_baseline.json`. No dataset download is needed at runtime.

## 2. Secrets

| Secret | Needed by | Where it lives | Never in |
|---|---|---|---|
| **None at runtime** | the app reads no secrets | n/a | n/a |
| `HF_TOKEN` (write scope, fine-grained to this one Space) | pushing the Space repository from this session | environment variable for the deploy command only, never written to disk in the repo | Git, source code, Dockerfile, logs, the Space's own variables |

- The deploy step pushes with a token supplied for that command, e.g. `git push https://user:$HF_TOKEN@huggingface.co/spaces/<owner>/<space>`. The remote URL is not saved with the token: `git -c credential.helper= push ...`, or the `huggingface_hub` upload API.
- Space **Variables** (non-secret) hold the `PDEFLOW_*` settings above. Space **Secrets** are not needed.

## 3. What the owner must provide at deployment time
1. The HF username or organisation, and the Space name (e.g. `<owner>/pdeflow`).
2. Either a write token scoped to that Space, given to this session as an environment variable, not pasted into a file; or the owner creates the Space and adds this session's push access.
3. Explicit authorization to deploy.

## 4. Verification after deploy (M6)
- `GET https://<owner>-<space>.hf.space/api/health` returns the pinned `checkpoint_sha256`.
- `/api/predict` on `test-4701` (fno128_native) reproduces the reported worst-case rel-L2, 0.24405, within 1e-6.
- Frontend loads; latency is recorded.
