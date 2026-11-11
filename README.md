# PDEFlow: PDE Solutions as a Service using Fourier Neural Operators

Learns the solution operator a ↦ u of 2D steady Darcy flow from the PDEBench Darcy Flow dataset (β = 1.0) with a
Fourier Neural Operator. The model will be served through a FastAPI backend and a Next.js + Plotly frontend.

**Status: M1 (data) complete. No model has been trained yet, and no model results exist.**

| Document | Contents |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | architecture, decision log, risks, milestones |
| [`docs/BLOCKERS.md`](docs/BLOCKERS.md) | environment/network assessment, measured CPU cost, schedule |
| [`docs/DATA.md`](docs/DATA.md) | dataset provenance, structure, physics verification, the two targets, data findings |
| [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) | splits, normalisation, 128→64 downsampling validation, evaluation plan |
| [`docs/M1_AUDIT_REPORT.md`](docs/M1_AUDIT_REPORT.md) | M1 scientific audit + reproducibility report |
| [`CHECKLIST.md`](CHECKLIST.md) | progress |

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
