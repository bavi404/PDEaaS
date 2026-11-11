# PDEFlow: PDE Solutions as a Service using Fourier Neural Operators

Learns the solution operator a ↦ u of 2D steady Darcy flow from the PDEBench Darcy Flow dataset (β = 1.0) with a
Fourier Neural Operator, served through a FastAPI backend and a Next.js + TypeScript + Plotly frontend.

**Status: M1 (data) and M2/M3 (baseline FNO training and evaluation) complete.** Test mean relative L2 vs the official
PDEBench targets is 0.0293 at 64×64 and 0.0421 at 128×128 zero-shot; see [`docs/RESULTS.md`](docs/RESULTS.md).
The FastAPI backend (M4) serves the evaluated checkpoint ([`docs/API.md`](docs/API.md)).
The Next.js + Plotly frontend (M5) runs a complete prediction workflow against the API ([`docs/FRONTEND.md`](docs/FRONTEND.md)).
Deployment is not done yet; the Hugging Face Spaces plan is in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

| Document | Contents |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | architecture, decision log, risks, milestones |
| [`docs/BLOCKERS.md`](docs/BLOCKERS.md) | environment/network assessment, measured CPU cost, schedule |
| [`docs/DATA.md`](docs/DATA.md) | dataset provenance, structure, physics verification, the two targets, data findings |
| [`docs/TRAINING.md`](docs/TRAINING.md) | model/training decisions, checkpoint format, run log notes |
| [`docs/RESULTS.md`](docs/RESULTS.md) | baseline test results (64² and 128²; official vs reference), latency, limitations |
| [`docs/API.md`](docs/API.md) | API endpoints, validation, logging, model-identity checks, latency |
| [`docs/FRONTEND.md`](docs/FRONTEND.md) | frontend architecture, pages, scientific presentation rules, tests, screenshots |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Hugging Face Spaces configuration and secrets (not deployed) |
| [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) | splits, normalisation, 128→64 downsampling validation, evaluation plan |
| [`docs/M1_AUDIT_REPORT.md`](docs/M1_AUDIT_REPORT.md) | M1 scientific audit + reproducibility report |
| [`CHECKLIST.md`](CHECKLIST.md) | progress |

## Run the application locally

Requires the Python environment below (`.venv`) and Node 22. No dataset is needed: the API serves the committed checkpoint
(SHA-256 pinned) and 24 bundled held-out test samples.

```bash
.venv/bin/uvicorn pdeflow.api.app:app --app-dir backend --host 127.0.0.1 --port 8000
```

```bash
cd frontend && npm ci && npm run build
PORT=3000 PDEFLOW_API_URL=http://127.0.0.1:8000 node .next/standalone/server.js   # open http://localhost:3000
```

![Results page: benchmark example compared with the official PDEBench target](docs/screenshots/desktop-3-results-benchmark.png)

Tests: `cd backend && ../.venv/bin/python -m pytest` (backend); `cd frontend && npm test && npm run e2e` (frontend; see docs/FRONTEND.md).

## Reproduce M1 (data pipeline)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -r backend/requirements.lock
uv pip install --python .venv/bin/python --no-deps -e backend
```

```bash
bash scripts/reproduce_m1.sh   # download+MD5, inspect, preprocess, downsampling validation, audit, solver verification, strict tests
```

The lock file pins the CPU-only torch wheel for Linux x86_64. On macOS, install `torch==2.15.0` from PyPI instead.
