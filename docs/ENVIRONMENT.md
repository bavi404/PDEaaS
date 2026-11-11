# Environment Discovery Log

Recorded at the start of the session (2026-11-11, ~07:23 UTC). Every value below
was observed directly in this container; nothing is assumed.

## Development container (Claude Code cloud session)

| Item | Observed |
|---|---|
| OS / arch | Ubuntu 24.04.5 LTS, x86_64 |
| CPU / RAM | 4 vCPU, 15 GiB RAM, no swap |
| Disk | ~31 GB free (per-session allowance) |
| GPU / CUDA | **None** — no `nvidia-smi`, no `/dev/nvidia*` |
| Python | 3.13.18 (`uv` available) |
| Pre-installed Py pkgs | numpy 2.5.3, pydantic 2.13.5, uvicorn 0.54.0, pandas, pillow, onnxruntime, httpx2, starlette |
| Missing Py pkgs | torch, h5py, fastapi, pytest, neuraloperator, scipy, matplotlib |
| Node / npm / pnpm | v22.23.4 / 10.9.9 / 10.28.0 |
| Docker | CLI 29.9.2, Compose v5.6.1; daemon not running by default, `dockerd` binary present and starts |
| Git | git works; `git ls-remote` to github.com succeeds (read) |
| GitHub auth | `gh auth status`: **token in GH_TOKEN is invalid** → push not possible yet |

## Network reachability (through the session's egress proxy)

| Host | Result | Needed for |
|---|---|---|
| pypi.org, files.pythonhosted.org | **DNS failure** | torch, h5py, fastapi, pytest |
| registry.npmjs.org | **DNS failure** | Next.js, Plotly, test tooling |
| download.pytorch.org | **403 (policy denied)** | CPU-only torch wheels |
| darus.uni-stuttgart.de | **403 (policy denied)** | PDEBench Darcy Flow data |
| raw.githubusercontent.com | **403 (policy denied)** | PDEBench URL manifest |
| huggingface.co | **403 (policy denied)** | optional deployment target |
| registry-1.docker.io | DNS failure | Docker base images |
| github.com / api.github.com | reachable | source, pushing (once auth fixed) |

## Target training hardware (user-provided, not visible from this container)

- NVIDIA RTX 5070, 8 GB VRAM (user's machine; reachable only if attached via Remote Control)
- MacBook Air, Dell Vostro

## Consequences

1. Nothing dependent on torch, h5py, FastAPI, or Next.js can be installed until
   the network policy allows package registries.
2. Real PDEBench data cannot be fetched until `darus.uni-stuttgart.de` is allowed.
   Per the project constraints we will **not** substitute synthetic data.
3. Training must be designed for CPU (this container) with an optional CUDA path.
