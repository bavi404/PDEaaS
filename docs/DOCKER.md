# Docker image (M6)

One image runs the whole application, compatible with a Hugging Face Docker Space:

```
                    container (UID 1000)
  :7860  ──▶  Next.js standalone server (node 22.23.4)  ──/api/*──▶  FastAPI / uvicorn 127.0.0.1:8000
  public       pages + same-origin API proxy                          FNO checkpoint (SHA-256 pinned)
                                                                      24 demo samples, evaluation report
```

The files are:
- `Dockerfile`
- `.dockerignore`: an allow-list, so the build context holds only what the image needs
- `docker/start.sh`: the entrypoint
- `backend/requirements-runtime.lock`
- `scripts/container_check.py`, `scripts/container_lifecycle.sh`: the container tests

## 1. Build and run

```bash
docker build -t pdeflow .
```

```bash
docker run --rm -p 7860:7860 pdeflow        # open http://localhost:7860
```

`docker compose up --build` does the same, with HF CPU-basic limits (2 CPUs, 16 GB) from `docker-compose.yml`. Verified healthy.

**Behind a TLS-intercepting proxy, or when Docker Hub is rate-limited** (this development sandbox). These options are not needed elsewhere:

```bash
DOCKER_BUILDKIT=1 docker build --network host \
  --build-arg REGISTRY=mirror.gcr.io/library \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" --build-arg HTTP_PROXY="$HTTPS_PROXY" \
  --secret id=ca_bundle,src=/path/to/ca-bundle.crt \
  -t pdeflow .
```

- **Mirror:** `mirror.gcr.io` serves the same image digests as Docker Hub, and the digests are pinned in the `Dockerfile`.
- **Proxy settings:** proxy build args are Docker's predefined proxy arguments, which are not recorded in image history.
- **CA bundle:** it is mounted as a BuildKit **secret**, used only inside the two `RUN` steps that download packages, and never written to a layer.
- **Verification:** `docker history --no-trunc` and `docker inspect` of the built image contain no proxy or CA reference.

## 2. Design decisions

| Decision | Alternatives | Justification |
|---|---|---|
| **One container, two processes** (Next.js public on 7860, FastAPI internal on 127.0.0.1:8000) | two containers + compose; FastAPI serving a static export | A HF Docker Space runs **one** container exposing one port. The frontend needs its server-side proxy route (runtime backend URL, 502 handling), so a static export is not enough. Binding FastAPI to localhost means it is reachable only through the proxy's allow-list. |
| **Three-stage build:** `node` → `web-build` (npm ci, `next build`, standalone output), `py-build` (venv from the runtime lock), `runtime` | single stage | The final image has no npm, no compilers, no frontend dev dependencies, no pip cache. It gets only the venv, the node binary and the 77 MB standalone server. |
| Base images pinned by **tag and digest**: `python:3.11.18-slim-trixie@sha256:07d4…`, `node:22.23.4-trixie-slim@sha256:be1e…` | floating tags | Reproducible. Python 3.11 as required (D13). Both images are Debian 13, so the copied `node` binary links against the same glibc / libstdc++. A build-time `node --version` checks this. |
| `backend/requirements-runtime.lock`: compiled from `requirements-runtime.in` **with `-c backend/requirements.lock`**, installed with `--no-deps`, then `pip check` | reuse the full dev lock | Every package has exactly the version tested in development: torch 2.15.0+cpu (CPU wheel index), neuraloperator 2.0.0, numpy 2.4.6, fastapi 0.162.4, uvicorn 0.54.0. No pytest or ruff. `neuraloperator` 2.0.0 declares h5py, matplotlib, scipy and wandb as hard dependencies, so they stay, honestly resolved. |
| Bash entrypoint as PID 1 with explicit `trap` | tini / supervisord | No extra packages, which avoids apt in the build. Startup ordering:
  1. verify the checkpoint SHA-256 (exit 3 on mismatch)
  2. start the API
  3. wait for `/health` (exit 1 if not ready in 60 s)
  4. start the web server

  SIGTERM/SIGINT go to both processes. If either dies, the other is stopped and the container exits non-zero, so the platform restarts it. All of this is tested in §4. |
| Checkpoint verified **three times**: at build (`sha256sum -c`), by the entrypoint, and by the API on load | once | The image cannot be built, started or served with a checkpoint other than the evaluated one (`d7133c7c…124df7`). |
| Non-root **UID 1000**; `/app` owned by root and read-only to the app | root | HF Docker Spaces run as UID 1000. The app writes nothing to disk; logs go to stderr. |
| `HEALTHCHECK` on `http://127.0.0.1:7860/api/health` | API port only | It checks the public server **and** the API behind it, because the proxy returns 502 when FastAPI is down. |
| `PDEFLOW_NUM_THREADS=2`, `OMP_NUM_THREADS=2` | 4 | Matches HF CPU basic (2 vCPU). One model forward runs at a time (an API lock), so the CPUs are not oversubscribed. |

**Image contents**, checked with `find` / `du` inside the image:

| Path | Size | Contents |
|---|---|---|
| `/opt/venv` | 1.2 GB | runtime Python packages (torch 747 MB) |
| `/usr/local/bin/node` | 120 MB | Node 22.23.4 binary |
| `/app/web` | 77 MB | Next.js standalone server + static assets |
| `/app/backend` | 5.5 MB | `pdeflow` package, checkpoint (2.8 MB), demo samples (2.8 MB) |
| `/app/reports` | 24 KB | `m2_eval_fno_baseline.json`, which `/model` serves only if its checkpoint hash matches |

- **Totals:** 2.04 GB on disk, **460 MB compressed**.
- **Not included:** the 1.3 GB PDEBench file, `data/processed`, `runs/`, logs, tests, docs, `.venv`, `node_modules`, `.git`, any `.env` or credentials. The only `.h5` files in the image are h5py's own tiny test fixtures inside site-packages.

## 3. Runtime configuration (environment variables; none are secrets)

| Variable | Image default | Meaning |
|---|---|---|
| `PORT` | 7860 | public port (Next.js) |
| `HOSTNAME` | 0.0.0.0 | Next.js bind address |
| `PDEFLOW_API_PORT` | 8000 | internal API port (bound to 127.0.0.1) |
| `PDEFLOW_NUM_THREADS` | 2 | torch intra-op threads |
| `PDEFLOW_CORS_ORIGINS` | empty | none needed: the browser only talks to the same origin |
| `PDEFLOW_MAX_BODY_BYTES` | 1000000 | request size limit |
| `PDEFLOW_EXPECTED_CHECKPOINT_SHA256` | pinned in code | override only to serve a different, deliberately re-evaluated checkpoint |
| `PDEFLOW_LOG_FILE` | unset | optional file copy of the JSON logs; by default stderr only (`docker logs`) |

## 4. Container test results

Image `pdeflow:m6`: build log in `reports/m6_docker_build.txt`, cold build 2 min 30 s on 4 vCPU. The container was run with HF CPU-basic limits: `--cpus=2 --memory=16g`.

| Test | Result | Evidence |
|---|---|---|
| Startup to public `/api/health` 200 | **8.4–8.8 s** (2 vCPU) | `reports/m6_container_lifecycle.txt` |
| Docker `HEALTHCHECK` | healthy | `reports/m6_container_check.json` |
| Functional checks through port 7860 | **17/17 pass**:
  - frontend loads; health; served SHA = evaluated SHA; model card; 24 examples
  - sample 4701 at 128² native reproduces the report's worst case (0.244054, rel 1e-6)
  - 64² sample with both targets kept separate
  - custom 128² field: no targets, no metrics
  - OOD warning; malformed → 422 with request id; invalid JSON → 422; unknown sample → 404
  - 32 concurrent predictions; structured request logs; memory; healthcheck | `reports/m6_container_check.{json,txt}` |
| Playwright e2e **against the container** (desktop + mobile) | **26/26 pass**:
  - prediction workflows, heatmaps with hover, no fabricated ground truth, OOD warnings, malformed input
  - API-unavailable handling: the image's web server with its API URL pointed at a closed port, plus a failure during prediction
  - no console errors, no horizontal overflow | `reports/m6_container_e2e.txt` |
| Request logs | 216 JSON `request` events (method, path, status, duration_ms, request_id) and 110 `prediction` events in `docker logs` | `container_check` |
| Graceful stop (`docker stop`) | exit 0 in **0.36 s**; uvicorn lifespan `shutdown` and entrypoint `stopped` are logged | lifecycle |
| Tampered checkpoint (bind-mounted random file) | refused before start: **exit 3**, `ERROR checkpoint_sha256_mismatch` | lifecycle |
| API (uvicorn) or web (next-server) killed | container stops, **exit 137**, `ERROR process_exited` | lifecycle |

**Resource usage**, measured during the e2e suite plus a burst of 32 concurrent 128² requests:

| Metric | Value |
|---|---|
| Container memory, cgroup peak | **400 MiB** (first run: 434 MiB). Limit 16 GiB. |
| Per-process RSS | uvicorn 432 MiB, next-server 141 MiB (RSS counts shared library pages per process) |
| CPU | peak 141 % of the 200 % allowed (2 vCPU), during the concurrent burst |
| Latency through port 7860, 20 sequential requests | 64² sample: median 26.5 ms. 128² sample (default path): 32.5 ms. 128² custom field: 37.9 ms. Server-side inference 13.5–16.1 ms. |
| Disk | image 2.04 GB; nothing written at runtime |

Rerun everything:

```bash
docker run -d --name pdeflow-m6 --cpus=2 --memory=16g -p 7860:7860 pdeflow
```

```bash
docker run -d --name pdeflow-m6-noapi -p 7861:7860 -e PDEFLOW_API_URL=http://127.0.0.1:9 --entrypoint node pdeflow /app/web/server.js
```

```bash
cd frontend && E2E_BASE_URL=http://127.0.0.1:7860 E2E_DEAD_URL=http://127.0.0.1:7861 npx playwright test --trace off && cd ..
```

```bash
.venv/bin/python scripts/container_check.py --url http://127.0.0.1:7860 --container pdeflow-m6
```

```bash
scripts/container_lifecycle.sh pdeflow
```
