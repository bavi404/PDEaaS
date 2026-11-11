# Licensing notice

| Material | Licence | Location |
|---|---|---|
| PDEFlow source code and documentation: Python package, scripts, frontend, Docker/deploy configuration, tests, docs | **MIT** (see [`LICENSE`](LICENSE)) | `backend/pdeflow/`, `backend/tests/`, `scripts/`, `frontend/`, `docker/`, `deploy/`, `docs/`, configs |
| Trained FNO checkpoint, produced by this project | **MIT**, as part of this project's own work. It was trained on PDEBench data (CC BY 4.0); attribution below. | `backend/models/fno_darcy_beta1_baseline.pt` |
| **PDEBench 2D Darcy Flow data**: the 24 bundled test samples (permeability and official targets) | **CC BY 4.0**, *not* MIT. © the PDEBench authors. | `backend/assets/demo_samples_v1.npz` |
| Independent steady-state reference fields for those samples | CC BY 4.0, *not* MIT. They are computed by this project from the PDEBench inputs, so they are an adaptation and keep the source licence. | `backend/assets/demo_samples_v1.npz` (`u_reference`) |
| Third-party software: PyTorch, neuraloperator, NumPy, FastAPI, Next.js, React, Plotly and other dependencies | **their own licences**, *not* relicensed by this project | installed from PyPI / npm / the PyTorch index at build time; versions in `backend/requirements*.lock`, `frontend/package-lock.json` |

## PDEBench attribution

Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflüger, Mathias Niepert.
*PDEBench: An Extensive Benchmark for Scientific Machine Learning.* NeurIPS 2022 Datasets and Benchmarks Track.
Data: *PDEBench Datasets*, DaRUS, doi:[10.18419/darus-2986](https://doi.org/10.18419/darus-2986), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). File used: `2D_DarcyFlow_beta1.0_Train.hdf5`; changes are
described in `docs/DATA.md` and `docs/PREPROCESSING.md`. The repository does not redistribute the full dataset.
