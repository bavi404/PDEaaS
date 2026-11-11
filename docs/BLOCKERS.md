# Blocker Assessment (revision 1)

Written 2026-11-11 ~07:40 UTC, about 17 min after session start (07:23 UTC).
Deadline: 2026-11-14 07:23 UTC. Every claim below is backed by a command run in this
session; where something could not be verified, that is stated.

## 0. Summary

| Area | Status | Who resolves it |
|---|---|---|
| Network policy change without a new session | **Resolved.** Applied to this same session | — |
| PyPI / npm access | **Resolved in-session** (proxy workaround, §2) | me |
| PyTorch CPU index | **Resolved** | — |
| PDEBench via official DaRUS mechanism | **Resolved**; range requests verified | — |
| Python 3.11 + package compatibility | **Resolved**: install and import verified | me |
| Docker pull, run and build-time network | **Resolved** with build-time proxy args (§2.4) | me |
| CPU-only training feasibility | **Feasible**: measured, see §4 | — |
| **GitHub push** | **Blocked**: `GH_TOKEN` invalid | **user** |
| **Public deployment credentials** | **Not yet provided** | **user** |
| GPU in this container | Not available (hardware, not policy). **Not required** | — |

## 1. Session continuity (single-session requirement)

**Question:** can network settings change without a new Claude Code session?
**Observed answer: yes.**

| Evidence | Value |
|---|---|
| Session ID (`get_session`) | `session_01VVFeg8GRCZTkpCCJEF8f3F`, unchanged |
| Local transcript | `~/.claude/projects/-home-user/b3e55451-b923-56d6-bd77-1ce2e63bbdef.jsonl`; first record `2026-11-11T07:23:24.850Z` holds the original prompt. Read only, never written. |
| Container boot time (`uptime -s`) | `2026-11-11 07:33:30`, i.e. **after** the first turn ended |
| `worker_epoch` (session metadata) | `2`: the worker was restarted once, presumably when the environment was edited |
| Repo after restart | `/home/user/pdeflow` intact, commit `7c11268` present |
| Hosts denied before / reachable after | download.pytorch.org 403→200, darus.uni-stuttgart.de 403→200, raw.githubusercontent.com 403→200, Docker Hub DNS-fail→reachable, huggingface.co 403→200 |

**Conclusion:** the environment edit restarted the container under the **same session**. The original
session ID, the full JSONL history and the working directory were all preserved. No new session was
created and none is needed.

**Residual risk (not verifiable from inside):** this is one observed restart, not a platform guarantee.
The repository exists only on this container's disk until it is pushed. **Pushing to GitHub is
therefore the most urgent item** (§5).

## 2. Network: what works and how

| Host | Result now | Notes |
|---|---|---|
| pypi.org, files.pythonhosted.org | 200 via proxy | Listed in `NO_PROXY`, so tools try a direct connection, and direct DNS resolution fails. **Fix: clear `NO_PROXY`/`no_proxy` for pip/uv/npm commands.** No settings change needed. |
| registry.npmjs.org | 200 via proxy | Same fix |
| download.pytorch.org/whl/cpu | 200 | `torch-2.15.0+cpu` wheels for cp311 and cp313 present |
| darus.uni-stuttgart.de | 200 | Official Dataverse API |
| s3.tik.uni-stuttgart.de | 206 | DaRUS redirects file downloads here (HTTP 303) |
| raw.githubusercontent.com | 200 | PDEBench URL manifest |
| registry-1.docker.io, auth.docker.io | 401 / 200 | Normal; `docker pull python:3.11-slim` succeeded |
| huggingface.co | 200 | Candidate deployment host |
| github.com | reachable | Push blocked by credentials, not network |

### 2.4 Docker specifics (verified)
- `dockerd` starts manually (overlayfs, cgroup v1) and is not running by default. `docker run` works.
- Containers **cannot** reach package indexes by default (`pip download` failed). With
  `--network host`, `HTTPS_PROXY` and the CA bundle mounted, it succeeded.
  → Dockerfiles will take optional proxy/CA **build args used only in the build stage** of this environment.
  They will not be baked into images, so the same Dockerfiles work unchanged on a normal machine or host.

## 3. Software stack compatibility (installed and imported, not just resolved)

Throwaway venvs in the scratchpad (outside the repo), `uv pip install` from the PyTorch CPU index + PyPI:

| Package | Version | Py 3.11.15 | Py 3.13.18 |
|---|---|---|---|
| torch | 2.15.0+cpu | ✅ import | ✅ import |
| neuraloperator | 2.0.0 | ✅ `FNO` forward at 64² and 128² | ✅ forward at 64² |
| h5py | 3.16.0 | ✅ | ✅ |
| fastapi | 0.162.4 | ✅ | ✅ |
| pytest | 9.2.0 | ✅ | ✅ |
| numpy (resolved) | 2.4.6 | ✅ | ✅ |

- Python 3.11 was already on the system (`/usr/bin/python3.11`, 3.11.18). uv also installed a managed
  3.11.15 in 2 s under `~/.local/share/uv/python`. System Python 3.13 was not touched.
- `neuraloperator` 2.0.0 pulls `wandb`, `tensorly`, `tensorly-torch` and `zencfg`. The full venv is 1.2 GB.

**Decision D13: Python 3.11 for the project.**
- Alternatives: 3.13 (system default) or 3.12.
- Both 3.11 and 3.13 passed. 3.11 matches the `python:3.11-slim` Docker base already verified here, so
  dev, test and container use one interpreter. It is also the most widely supported version for the
  scientific-Python stack.

**D5 is resolved: use `neuraloperator.models.FNO`.** It installs and runs, so the in-house fallback is not
needed. The `build_model(cfg)` factory stays as the single entry point.

## 4. PDEBench Darcy Flow: provenance, size, access

From the DaRUS Dataverse API, dataset `doi:10.18419/darus-2986`, directory `2D/DarcyFlow`:

| DaRUS file id | File | Size | MD5 (from DaRUS) | Access |
|---|---|---|---|---|
| 133217 | 2D_DarcyFlow_beta0.01_Train.hdf5 | 1.311 GB | d05c287d4c0b7d3178b0097084238251 | public |
| 133218 | 2D_DarcyFlow_beta0.1_Train.hdf5 | 1.311 GB | 294f9a03a4aa16b0e386469ca8b471be | public |
| **133219** | **2D_DarcyFlow_beta1.0_Train.hdf5** | **1.311 GB** (1,310,724,488 B) | **81694ed31306ff2e5f6b76349b0b4389** | public |
| 133220 | 2D_DarcyFlow_beta10.0_Train.hdf5 | 1.311 GB | a7f23cf8011fc211b180828af39b7d1a | public |
| 133221 | 2D_DarcyFlow_beta100.0_Train.hdf5 | 1.311 GB | 7c09f3b1bb097737d3fd52ffb0c6f1c8 | public |

Official mechanism checked: `GET https://darus.uni-stuttgart.de/api/access/datafile/{id}` (what PDEBench's
download script uses) → HTTP 303 to a pre-signed `s3.tik.uni-stuttgart.de` URL.

| Test | Result |
|---|---|
| `HEAD` on the pre-signed URL | **403**: signature valid for GET only; not a real obstacle |
| `GET` with `Range: bytes=0-1023` | **206 Partial Content**, `content-range: bytes 0-1023/1310724488` |
| First 8 bytes | `\x89HDF\r\n\x1a\n`: a genuine HDF5 signature |
| Throughput (100 MB range GET) | 104,857,600 B in 7.9 s ≈ **13 MB/s** |

**Decision: download the full β = 1.0 file (~100 s at the measured rate) and verify its MD5 against
DaRUS.** Range requests are supported, but partial HDF5 reads would also depend on the file's internal
chunk layout, which is not yet inspected. The partial approach would add complexity and weaken provenance
(no whole-file checksum) to save about 100 s. Not worth it.

Still **unverified until download** (not assumed): HDF5 dataset names, shapes, dtypes, sample count,
coordinate grids. These go into `docs/DATA.md`.

## 5. CPU training feasibility (measured)

Benchmark: `docs/benchmarks/cpu_step_benchmark.py`; raw output: `cpu_step_benchmark_results.json`.
- Setup: `neuraloperator` FNO, AdamW, relative-L2 loss, batch 16, 4 torch threads, **random tensors**.
- These numbers are timing measurements only, **not** training results.
- Per config: 2 warm-up steps, then the mean of 5 steps. Inference is the mean of 20 forward passes at batch 1.

| Config | Params | Train step (b=16) | Per sample | Inference b=1 |
|---|---|---|---|---|
| A: 64², width 32, 12 modes, 4 layers | 357,217 | 0.240 s | **15.0 ms** | 13.2 ms |
| B: 128², width 32, 12 modes, 4 layers | 357,217 | 1.929 s | 120.5 ms | 20.1 ms |
| C: 64², width 64, 16 modes, 4 layers | 2,410,177 | 0.495 s | 31.0 ms | 24.0 ms |
| D: 32², width 32, 12 modes, 4 layers | 357,217 | 0.079 s | 4.9 ms | 8.0 ms |

Peak process RSS was 812 MB after config A. Later rows report the process-wide maximum, which config B
set at 1,630 MB, so their RSS is not attributable to them individually.

### Memory estimate (analytic)
| Item | Size |
|---|---|
| Full β file on disk | 1.31 GB (consistent with 2 × 10,000 × 128 × 128 × 4 B ≈ 1.31 GB if it holds 10k float32 pairs; to be confirmed) |
| Subset of 2,500 samples (input + target) at 128² float32 | 2,500 × 16,384 × 4 B × 2 ≈ **328 MB** |
| Same subset at 64² | ≈ **82 MB** |
| Training process (config A) | ≈ 0.8 GB measured |
| **Total RAM needed** | **< 2 GB** of the 15 GB available |

### Runtime estimate (from measured per-sample cost, plus ~15% for validation)

| Plan | Train samples | Epochs | Estimated wall time |
|---|---|---|---|
| **A @ 64², N_train = 2,000** (recommended) | 2,000 | 100 | **≈ 0.9 h** |
| A @ 64², N_train = 2,000 | 2,000 | 200 | ≈ 1.9 h |
| A @ 64², N_train = 8,000 | 8,000 | 100 | ≈ 3.8 h |
| C @ 64² (ablation, larger model) | 2,000 | 100 | ≈ 2.0 h |
| B @ 128² | 2,000 | 100 | ≈ 7.7 h (not recommended) |

### Scientific justification for the reductions
- **64² training via stride-2 subsampling of the 128² PDEBench grid.** FNO is formulated as a
  discretisation-invariant operator, and the original FNO Darcy experiments trained on subsampled grids.
  Evaluating the trained model on the native 128² test inputs tests this directly. We will report the
  actual 64² and 128² errors whatever they turn out to be. 128² training costs ~8× more (measured).
- **N_train ≈ 2,000 of 10,000.** This is in the same range as standard FNO Darcy setups (about 1,000
  training samples). The original plan already specified it (D1), so it is unchanged. Splits stay 80/10/10
  of a seeded 2,500-sample draw (2,000 / 250 / 250). If time permits, a larger-N run is an optional extension.
- **Model A (357k params) as the main model, C as an optional ablation.** Both fit comfortably in the budget.
- **GPU not required.** The RTX 5070 would cut training time but is not needed to meet the deadline, and it
  would add a cross-machine dependency. Not pursued unless runs prove slower than measured.

### Minimum resources to complete the project
- **Compute:** 4 vCPU, ≥ 4 GB RAM (training needs < 2 GB), no GPU.
- **Disk:** ≈ 6–8 GB (1.3 GB data, ≈ 1.2 GB venv, ≈ 3–5 GB Docker images / node_modules). 31 GB available.
- **CPU wall time:** ≈ 1 h for the main training run, plus ≈ 2 h optional ablation, plus minutes for evaluation.
- **Network:** PyPI, PyTorch CPU index, npm, DaRUS + `s3.tik.uni-stuttgart.de`, Docker Hub, GitHub (push),
  and the chosen deployment host. All are reachable now except GitHub push, which is blocked by credentials.

## 6. What remains

### 6.1 Resolvable within this session (by me)
- Run pip, uv and npm with `NO_PROXY` cleared, and start `dockerd` manually.
- Create the project venv (Python 3.11), pin dependencies, download and MD5-verify the β = 1.0 file.
- Everything in milestones M1–M5, plus the Compose build and E2E verification.

### 6.2 Requires your intervention
1. **GitHub push (urgent).** `gh auth status` reports the `GH_TOKEN` as invalid, so I can neither push nor
   create a repository.
   - Create an empty repository, e.g. `pdeflow`.
   - Connect or reconnect GitHub at https://claude.ai/connect-github and install the Claude GitHub App on
     that repository.
   - Tell me `owner/repo`. I then attach it **to this same session** with `add_repo` (push access); no new
     session is needed.
2. **Deployment target and credentials.** huggingface.co is reachable. Hugging Face Spaces (Docker) is still
   the recommendation (D12). I need the Space name or account and an HF write token added as an environment
   secret. Adding a secret may restart the container as before; §1 shows this kept the session.
3. **Commit author identity**, carried over from earlier: keep "Claude" with the co-author trailer, or use
   your name and email.

### 6.3 Impossible under the current policy and environment
- **Using a GPU in this container:** no NVIDIA device exists (hardware, not network policy). Not needed.
- **Pushing or creating a GitHub repo with the current token:** impossible until it is replaced (§6.2).
- **Verifying the platform's long-term transcript retention from inside the container:** only the single
  restart in §1 is observable. I never write to the JSONL. For a safety copy, export the transcript from
  the claude.ai side.

Nothing in the project scope is blocked by the network policy any more.

## 7. Revised schedule (UTC, deadline 2026-11-14 07:23)

| M | Window (UTC) | Deliverable | Change vs plan |
|---|---|---|---|
| M0 | 11 Nov 07:23–08:00 | Discovery, plan, blocker assessment | +this document |
| M1 | 11 Nov 08:00–13:00 | Venv, data download + MD5, HDF5 inspection → `DATA.md`, splits, normalizer, tests | Shorter: download is ~2 min |
| M2 | 11 Nov 13:00–12 Nov 01:00 | FNO factory, training loop, checkpoints, **main run (~1 h)**, optional ablation (~2 h) | Same window, now with measured runtime |
| M3 | 12 Nov 01:00–09:00 | Test rel-L2 at 64² and 128², latency benchmark, `RESULTS.md` | — |
| M4 | 12 Nov 09:00–20:00 | FastAPI + validation + logging + pytest | — |
| M5 | 12 Nov 20:00–13 Nov 10:00 | Next.js + Plotly UI + tests | — |
| M6 | 13 Nov 10:00–20:00 | Dockerfiles, Compose E2E, public deployment | Depends on §6.2 item 2 |
| M7 | 13 Nov 20:00–14 Nov 07:23 | README, architecture doc, submission checklist, ~11 h buffer | — |

The GitHub push should happen **as soon as §6.2 item 1 is done**, then after every milestone.
