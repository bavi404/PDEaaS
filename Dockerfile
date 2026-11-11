#
# PDEFlow: single-container image (Hugging Face Docker Space compatible, port 7860).
#   Next.js frontend (standalone, production build)  -> 0.0.0.0:7860, public
#   FastAPI + FNO checkpoint (SHA-256 pinned)         -> 127.0.0.1:8000, internal only
# See docs/DOCKER.md for build/run instructions and the reasoning behind each choice.
#
# Base images are pinned by tag AND digest. REGISTRY defaults to Docker Hub; a mirror serving the same digests
# can be substituted at build time (--build-arg REGISTRY=mirror.gcr.io/library).
ARG REGISTRY=docker.io/library
ARG NODE_IMAGE=node:22.23.4-trixie-slim@sha256:be1e2a3dc2c4fd68cdfadad2a696a280e0d2749098bee509230174d72fb768c6
ARG PYTHON_IMAGE=python:3.11.18-slim-trixie@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

# ---------------------------------------------------------------------------------------------------------------
FROM ${REGISTRY}/${NODE_IMAGE} AS node

# ---------------------------------------------------------------------------------------------------------------
FROM node AS web-build
WORKDIR /src/frontend
ENV NEXT_TELEMETRY_DISABLED=1
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
# Optional CA bundle for TLS-intercepting build proxies; a BuildKit secret, never stored in a layer.
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca_bundle; fi; \
    npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------------------------------------------
FROM ${REGISTRY}/${PYTHON_IMAGE} AS py-build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY backend/requirements-runtime.lock /tmp/requirements-runtime.lock
# The lock is complete (generated with constraints from backend/requirements.lock), so --no-deps installs
# exactly the listed versions; the CPU-only torch wheel comes from the index declared inside the lock.
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export PIP_CERT=/run/secrets/ca_bundle; fi; \
    python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-deps -r /tmp/requirements-runtime.lock \
 && /opt/venv/bin/pip check \
 && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------------------------------------------
FROM ${REGISTRY}/${PYTHON_IMAGE} AS runtime

# Node runtime binary only (no npm); the standalone Next.js server bundles its own node_modules.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=py-build /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=7860 \
    HOSTNAME=0.0.0.0 \
    PDEFLOW_API_PORT=8000 \
    PDEFLOW_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    PDEFLOW_CORS_ORIGINS="" \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
# Application payload: API package, the evaluated checkpoint, 24 attributed demo samples (CC BY 4.0) and the
# evaluation report whose metrics /model serves. No dataset, runs, caches or tests.
COPY backend/pdeflow /app/backend/pdeflow
COPY backend/models/fno_darcy_beta1_baseline.pt /app/backend/models/fno_darcy_beta1_baseline.pt
COPY backend/assets/demo_samples_v1.npz /app/backend/assets/demo_samples_v1.npz
COPY reports/m2_eval_fno_baseline.json /app/reports/m2_eval_fno_baseline.json
COPY --from=web-build /src/frontend/.next/standalone /app/web
COPY docker/start.sh /app/start.sh

# Build-time integrity checks: the evaluated checkpoint, a working node binary, and an importable API.
RUN echo "d7133c7c681cf3e9632579833fe357636d84ff7a5678d94b2f24c38955124df7  /app/backend/models/fno_darcy_beta1_baseline.pt" | sha256sum -c - \
 && node --version \
 && python -c "import pdeflow.api.app" \
 && useradd --create-home --uid 1000 --shell /usr/sbin/nologin user \
 && chmod 0555 /app/start.sh

# Hugging Face Docker Spaces run as UID 1000; the app needs no write access to /app.
USER 1000
ENV HOME=/home/user
EXPOSE 7860

# Checks the public server AND the API behind it (the proxy returns 502 if FastAPI is down).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["/app/start.sh"]
