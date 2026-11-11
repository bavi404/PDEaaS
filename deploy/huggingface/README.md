---
title: PDEFlow
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FNO surrogate for 2D Darcy flow (PDEBench), FastAPI + Next.js
---

# PDEFlow: PDE solutions as a service

A Fourier Neural Operator, trained on the PDEBench 2D Darcy Flow benchmark (β = 1.0), served through FastAPI behind a
Next.js + Plotly interface. Give it a permeability field and it returns the steady pressure field.

- **Benchmark examples:** 24 held-out PDEBench test samples. Each prediction is compared with the official
  PDEBench target. An independent steady-state reference is shown separately, as secondary analysis.
- **Custom fields:** 64×64 or 128×128. Out-of-distribution inputs are flagged, and no ground truth or error
  metric is invented for them.
- **Checkpoint integrity:** the served checkpoint's SHA-256 is pinned and checked at startup.

**Data attribution:** PDEBench (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks), DaRUS
doi:10.18419/darus-2986, licensed CC BY 4.0. The Space bundles 24 test samples only.

**Licence:** MIT covers this project's source code (and its trained checkpoint). The bundled PDEBench samples stay
CC BY 4.0, and third-party components keep their own licences (NOTICE.md in the repository).

**Source and documentation:** the project's GitHub repository (README, docs/DOCKER.md, docs/DEPLOYMENT.md).
