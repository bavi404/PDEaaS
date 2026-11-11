# Deployment: Hugging Face Spaces (Docker SDK)

**Status: NOT deployed.** No public URL exists yet. The image is built and verified locally (docs/DOCKER.md §4).
Deployment requires the owner's authorization and a token; see §6.

## 1. Space configuration

The Space card is `deploy/huggingface/README.md`. It replaces `README.md` in the Space repository:

```yaml
---
title: PDEFlow
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: FNO surrogate for 2D Darcy flow (PDEBench), FastAPI + Next.js
---
```

| Setting | Value | Reason / measurement |
|---|---|---|
| SDK | Docker | The image runs Python 3.11 + torch CPU + Node in one container (docs/DOCKER.md) |
| Port | **7860** (`app_port`) | Next.js listens on 0.0.0.0:7860; FastAPI stays internal on 127.0.0.1:8000 |
| Hardware | **CPU basic** (free: 2 vCPU, 16 GB RAM) | Measured with `--cpus=2 --memory=16g`: peak **400–434 MiB** memory, peak CPU 141 % of 200 %, 128² request about 33–38 ms |
| Storage | about 0.5 GB compressed image (2.04 GB unpacked); no persistent storage | Nothing is written at runtime; activity is in memory |
| User | UID 1000 (set in the Dockerfile) | HF Docker Spaces requirement |
| Visibility | public | required by the brief |
| Licence field | **not set**: the owner should choose a code licence | Bundled PDEBench samples are CC BY 4.0, with attribution in the Space card and `/model` |

## 2. Environment variables

**None need to be set in the Space.** The image defaults are correct for CPU basic (docs/DOCKER.md §3):
`PORT=7860`, `PDEFLOW_NUM_THREADS=2`, `PDEFLOW_CORS_ORIGINS=` (same-origin).

If the hardware is upgraded, set `PDEFLOW_NUM_THREADS` and `OMP_NUM_THREADS` to the vCPU count as Space **Variables**; they are not secrets.

## 3. Secrets

| Secret | Used by | Where it lives | Never in |
|---|---|---|---|
| none | the running app | n/a: the app reads no secrets | n/a |
| `HF_TOKEN`: write access, ideally a fine-grained token scoped to this one Space | `scripts/deploy_hf_space.py`, at deploy time only | an environment variable for the deploy command, supplied by the owner | Git, source, Dockerfile, image, Space variables/secrets, logs. The script never prints it. |

## 4. Deployment command

The script uploads exactly the committed files the Docker build needs:
- the `.dockerignore` allow-list, taken from `git archive HEAD`
- the Space card, as `README.md`

It refuses to run with uncommitted changes. HF then builds the image from the Space repository.

```bash
# preview: lists the staged files, no network, no token
uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 python scripts/deploy_hf_space.py --space <owner>/<space> --dry-run
```

```bash
# deploy (owner-authorised; token only in the environment)
HF_TOKEN=... uv run --no-project --python 3.11 --with huggingface_hub==2.2.0 python scripts/deploy_hf_space.py --space <owner>/<space>
```

**Verified locally:**
- **Dry run:** `--dry-run --stage-to` staged **57 files (5.9 MB)** from the committed HEAD (`reports/m6_space_dry_run.txt`).
- **Image from the staged repository:** built with `docker build`; the 17 container checks pass (`reports/m6_space_context_check.json`), with peak memory 423 MiB.

**Equivalence with the tested image.** The staged directory is the same build context as the locally tested image. Two things differ on HF:
- No build args are given, so base images come from Docker Hub by the same pinned digests.
- No CA secret is given; the `required=false` secret mount is simply absent.

## 5. Expected startup behaviour

1. **HF builds the image from the Space repository.** Measured locally: 2 min 30 s on 4 vCPU; expect longer on HF builders.
2. **The container starts, and `docker/start.sh` logs JSON events to the Space's Logs tab:**
   - `checkpoint_verified`, carrying `sha256 d7133c7c…124df7`
   - API `startup` (24 samples, metrics loaded)
   - `api_ready`, then `web_started`

   This sequence took **about 8.5 s** with 2 vCPU locally.
3. **Failure modes:**
   - A checkpoint hash mismatch → `ERROR checkpoint_sha256_mismatch`, exit 3. The Space shows a runtime error, and nothing is served with an unverified model.
   - A crash of either process → `ERROR process_exited`, non-zero exit, and HF restarts the container.
4. **Public URL:** `https://<owner>-<space>.hf.space`. Free CPU-basic Spaces sleep after a period of inactivity and cold-start on the next visit (build cache kept; about 10 s app startup plus platform overhead).

## 6. Post-deployment verification (required before claiming deployment success)

```bash
URL=https://<owner>-<space>.hf.space
```

```bash
curl -s $URL/api/health          # checkpoint_sha256 must be d7133c7c...124df7
```

```bash
.venv/bin/python scripts/container_check.py --url $URL --container none --out reports/m7_deployment_check.json   # HTTP checks only
```

```bash
cd frontend && E2E_BASE_URL=$URL npx playwright test --trace off --grep-invert "unavailable"
```

These checks need a working public URL, so they can only run after deployment. The outcome will be recorded in `reports/m7_deployment_check.*`.

## 7. What is needed from the owner

1. The Space id `<owner>/<space>` (e.g. `bavi404/pdeflow`).
2. A write token in the `HF_TOKEN` environment variable of this session, not pasted into any file. Alternatively, the owner runs the command in §4 themselves.
3. Explicit authorization to deploy.
4. Optional: a code licence for the Space card.
