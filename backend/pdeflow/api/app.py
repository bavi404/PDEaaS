"""PDEFlow FastAPI service: FNO surrogate for steady 2D Darcy flow (PDEBench, beta = 1.0).

Endpoints
  GET  /health                 liveness + loaded checkpoint hash
  GET  /model                  model card: architecture, checkpoint, training config, measured test metrics
  GET  /samples                bundled held-out test samples
  GET  /samples/{id}           one sample: permeability + both targets (separately named)
  POST /predict                predict u for a permeability field or a bundled sample

    uvicorn pdeflow.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pdeflow import __version__
from pdeflow.api.schemas import (
    Activity,
    ActivityItem,
    Health,
    PredictRequest,
    PredictResponse,
    SampleDetail,
    SampleEvaluation,
    SampleSummary,
    Targets,
)
from pdeflow.api.settings import Settings
from pdeflow.evaluation.metrics import PRIMARY, SECONDARY, relative_l2
from pdeflow.inference import SUPPORTED_RESOLUTIONS, TRAINING_VALUES, Predictor
from pdeflow.logging_utils import get_logger, log_event

ACTIVITY_MAX = 100  # in-memory ring buffer of prediction metadata; resets on restart
_REQ_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_PERMEABILITY = 1e6  # hard sanity bound; anything above is almost certainly a unit/encoding error


class SampleStore:
    """Bundled test samples. Targets are kept in separately named arrays (never merged)."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.indices = z["indices"].astype(int)
            self.a = z["a"]
            self.u_official = z["u_official"]
            self.u_reference = z["u_reference"]
            self.residual = z["official_target_residual"].astype(float)
            self.meta = json.loads(str(z["meta"]))
        self.ids = [f"test-{i}" for i in self.indices]
        self._pos = {s: k for k, s in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def position(self, sample_id: str) -> int:
        if sample_id not in self._pos:
            raise HTTPException(404, f"unknown sample_id '{sample_id}'")
        return self._pos[sample_id]

    def summary(self, k: int) -> SampleSummary:
        return SampleSummary(id=self.ids[k], test_index=int(self.indices[k]),
                             fraction_high_permeability=float((self.a[k] == self.a[k].max()).mean()),
                             official_target_pde_residual=float(self.residual[k]))

    def fields(self, k: int, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = 128 // resolution  # same stride-2 convention as training/evaluation
        return (self.a[k, ::s, ::s].astype(np.float64), self.u_official[k, ::s, ::s].astype(np.float64),
                self.u_reference[k, ::s, ::s].astype(np.float64))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _grid(x: np.ndarray) -> list[list[float]]:
    return x.astype(np.float64).tolist()


def validate_permeability(rows: list[list[float]]) -> tuple[np.ndarray, list[str]]:
    """Shape/value checks. Returns the field and distribution-shift warnings (not errors)."""
    n = len(rows)
    if n not in SUPPORTED_RESOLUTIONS or any(len(r) != n for r in rows):
        raise HTTPException(422, f"permeability must be a square grid of size 64x64 or 128x128 "
                                 f"(measured resolutions); got {n} rows with lengths "
                                 f"{sorted({len(r) for r in rows})[:5]}")
    a = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(a).all():
        raise HTTPException(422, "permeability contains non-finite values")
    if (a <= 0).any():
        raise HTTPException(422, "permeability must be strictly positive")
    if (a > MAX_PERMEABILITY).any():
        raise HTTPException(422, f"permeability values above {MAX_PERMEABILITY:g} are rejected")
    warnings = []
    if a.min() < min(TRAINING_VALUES) or a.max() > max(TRAINING_VALUES):
        warnings.append(f"values outside the training range [{min(TRAINING_VALUES)}, {max(TRAINING_VALUES)}]: "
                        "extrapolation, accuracy unknown")
    if not np.isin(a.astype(np.float32), np.float32(TRAINING_VALUES)).all():
        warnings.append("non-binary field: training data contain only the values 0.1 and 1.0; accuracy unknown")
    return a, warnings


def _load_metrics(report: Path, sha: str, log: logging.Logger) -> dict | None:
    """Measured test metrics, only if the report belongs to the served checkpoint."""
    if not report.exists():
        log_event(log, "eval_report_missing", level=logging.WARNING, path=str(report))
        return None
    r = json.loads(report.read_text())
    if r.get("checkpoint_sha256") != sha:
        log_event(log, "eval_report_checkpoint_mismatch", level=logging.WARNING, path=str(report))
        return None
    pick = lambda b: {PRIMARY: b[PRIMARY]["relative_l2"], SECONDARY: b[SECONDARY]["relative_l2"]}
    return {"source": report.name, "n_test": r["n_test"], "metric": "per-sample relative L2",
            "res64_fno_native": pick(r["res64"]), "res128_fno128_native": pick(r["res128"]),
            "res128_fno64_interpolated": pick(r["res128_interpolation_baseline"]),
            "note": r["note"]}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    log = get_logger("pdeflow.api", settings.log_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        predictor = Predictor(settings.checkpoint, settings.expected_sha256, settings.num_threads)  # fails on hash mismatch
        app.state.predictor = predictor
        app.state.samples = SampleStore(settings.samples_file)
        app.state.metrics = _load_metrics(settings.eval_report, predictor.sha256, log)
        app.state.activity = deque(maxlen=ACTIVITY_MAX)
        app.state.activity_total = 0
        app.state.started_utc = _now()
        log_event(log, "startup", checkpoint=str(settings.checkpoint), sha256=predictor.sha256,
                  n_samples=len(app.state.samples), metrics_loaded=app.state.metrics is not None,
                  num_threads=settings.num_threads, load_s=round(time.perf_counter() - t0, 3))
        yield
        log_event(log, "shutdown")

    app = FastAPI(title="PDEFlow", version=__version__, lifespan=lifespan,
                  description="Fourier Neural Operator surrogate for 2D Darcy flow (PDEBench beta=1.0).")
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if _REQ_ID.match(rid) else uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        cl = request.headers.get("content-length")
        if cl is not None and (not cl.isdigit() or int(cl) > settings.max_body_bytes):
            resp = JSONResponse({"detail": f"request body exceeds {settings.max_body_bytes} bytes",
                                 "request_id": rid}, status_code=413)
        else:
            try:
                resp = await call_next(request)
            except Exception:
                log.exception("unhandled_error", extra={"fields": {"request_id": rid, "path": request.url.path}})
                resp = JSONResponse({"detail": "internal server error", "request_id": rid}, status_code=500)
        resp.headers["X-Request-ID"] = rid
        log_event(log, "request", request_id=rid, method=request.method, path=request.url.path,
                  status=resp.status_code, duration_ms=round((time.perf_counter() - t0) * 1000, 2))
        return resp

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
                            status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errs = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()[:10]]
        return JSONResponse({"detail": jsonable_encoder(errs), "request_id": getattr(request.state, "request_id", None)},
                            status_code=422)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        p: Predictor = request.app.state.predictor
        return Health(status="ok", model_loaded=True, checkpoint_sha256=p.sha256, n_samples=len(request.app.state.samples))

    @app.get("/model")
    def model_card(request: Request) -> dict:
        st = request.app.state
        return {"service_version": __version__, "checkpoint": st.predictor.info(),
                "dataset": st.samples.meta["attribution"],
                "targets": {"official_pdebench_target": "PRIMARY benchmark: the FNO is trained and selected on it",
                            "independent_steady_state_reference": "SECONDARY analysis only: exact steady state "
                            "computed by this project; never used for training or selection"},
                "test_metrics": st.metrics}

    @app.get("/samples", response_model=list[SampleSummary])
    def list_samples(request: Request) -> list[SampleSummary]:
        s: SampleStore = request.app.state.samples
        return [s.summary(k) for k in range(len(s))]

    @app.get("/samples/{sample_id}", response_model=SampleDetail)
    def get_sample(request: Request, sample_id: str, resolution: int = Query(128)) -> SampleDetail:
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise HTTPException(422, f"resolution must be one of {SUPPORTED_RESOLUTIONS}")
        s: SampleStore = request.app.state.samples
        k = s.position(sample_id)
        a, off, ref = s.fields(k, resolution)
        return SampleDetail(**s.summary(k).model_dump(), resolution=resolution, permeability=_grid(a),
                            targets=Targets(official_pdebench_target=_grid(off),
                                            independent_steady_state_reference=_grid(ref)))

    @app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
    def predict(request: Request, body: PredictRequest) -> PredictResponse:
        st = request.app.state
        p: Predictor = st.predictor
        off = ref = None
        if body.sample_id is not None:
            s: SampleStore = st.samples
            a, off, ref = s.fields(s.position(body.sample_id), body.resolution)
            warnings: list[str] = []
        else:
            a, warnings = validate_permeability(body.permeability)
        try:
            pred = p.predict(a, body.strategy)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        u = pred.u
        resp = PredictResponse(
            prediction=_grid(u), resolution=pred.resolution, model_resolution=pred.model_resolution,
            strategy=pred.strategy, inference_ms=round(pred.inference_ms, 3), checkpoint_sha256=p.sha256,
            warnings=warnings, sample_id=body.sample_id,
            input_kind="benchmark_example" if body.sample_id is not None else "custom",
            input_summary={"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                           "fraction_at_max": float((a == a.max()).mean())})
        if off is not None:
            def err(x, t):
                return {"relative_l2": float(relative_l2(x[None], t[None])[0])}
            resp.evaluation = SampleEvaluation(primary_vs_official_pdebench_target=err(u, off),
                                               secondary_vs_independent_steady_state_reference=err(u, ref),
                                               official_target_vs_reference=err(off, ref))
            if body.return_targets:
                resp.targets = Targets(official_pdebench_target=_grid(off), independent_steady_state_reference=_grid(ref))
        log_event(log, "prediction", request_id=request.state.request_id, sample_id=body.sample_id,
                  resolution=pred.resolution, strategy=pred.strategy, inference_ms=round(pred.inference_ms, 3),
                  warnings=len(warnings),
                  rel_l2_official=resp.evaluation.primary_vs_official_pdebench_target.relative_l2 if resp.evaluation else None)
        st.activity.appendleft(ActivityItem(
            timestamp_utc=_now(), input_kind=resp.input_kind, sample_id=body.sample_id, resolution=pred.resolution,
            strategy=pred.strategy, inference_ms=resp.inference_ms, n_warnings=len(warnings),
            rel_l2_vs_official=resp.evaluation.primary_vs_official_pdebench_target.relative_l2 if resp.evaluation else None))
        st.activity_total += 1
        return resp

    @app.get("/activity", response_model=Activity)
    def activity(request: Request, limit: int = Query(10, ge=1, le=ACTIVITY_MAX)) -> Activity:
        st = request.app.state
        return Activity(items=list(st.activity)[:limit], total_since_start=st.activity_total,
                        started_utc=st.started_utc,
                        note="Predictions served by this API process since it started (metadata only, in memory).")

    return app


app = create_app()
