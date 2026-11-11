---
title: PDEFlow
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: app.py
pinned: false
license: mit
short_description: FNO surrogate for 2D Darcy flow (PDEBench), FastAPI + Gradio
---

# PDEFlow: PDE solutions as a service

A Fourier Neural Operator, trained on the PDEBench 2D Darcy Flow benchmark (β = 1.0), predicts the steady pressure
field u for a permeability field a, where −∇·(a∇u) = 1 on (0,1)² and u = 0 on the boundary.

| Path | What |
|---|---|
| `/` | web app (dashboard, solver workspace, results, model information) |
| `/gradio` | Gradio interface and Gradio API (benchmark examples and custom fields) |
| `/api/docs` | REST API (`/api/health`, `/api/model`, `/api/samples`, `/api/predict`, `/api/activity`) |

- **Benchmark examples:** 24 held-out PDEBench test samples. Each prediction is compared with the **official PDEBench
  target** (the primary metric). An independent steady-state reference is shown **separately**, as secondary analysis.
- **Custom fields:** 64×64 or 128×128. Out-of-distribution inputs are flagged, and no ground truth or error metric is
  invented for them.
- **Integrity:** the checkpoint's SHA-256 (`d7133c7c…124df7`) is pinned and verified at startup.
- **Hardware:** inference runs on CPU (about 15 ms per request); this Space requests no GPU time.

**Data attribution:** PDEBench (Takamoto, Praditia, Leiteritz, MacKinlay, Alesiani, Pflüger, Niepert; NeurIPS 2022
Datasets & Benchmarks), *PDEBench Datasets*, DaRUS doi:10.18419/darus-2986, licensed CC BY 4.0. This Space bundles
24 test samples only.

**Licence:** MIT for this project's code and trained checkpoint. The bundled PDEBench samples remain CC BY 4.0, and
third-party components keep their own licences. See `LICENSE` and `NOTICE.md`.
