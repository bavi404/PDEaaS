"""PDEFlow on a Hugging Face ZeroGPU Space (Gradio SDK): one process, three surfaces (docs/ZEROGPU.md).

    /api/*     the existing FastAPI app, `pdeflow.api.app.create_app()`, unchanged: pinned-SHA checkpoint, Predictor,
               validation, JSON logs
    /gradio    Gradio UI + Gradio API, calling the SAME Predictor and the SAME sample/validation code as /api
    /          the existing Next.js frontend as a static export (deploy/zerogpu/build_frontend.py); same-origin /api

Inference runs on CPU. No function is decorated with @spaces.GPU, so no GPU is requested and visitors' GPU quota is
never used. `spaces` is imported first, as ZeroGPU documents, and is effect-free elsewhere. It is optional so the
module also imports in the Python 3.11 dev environment.

Layouts:
  Space repo:  app.py, backend/pdeflow, backend/models, backend/assets, reports/, web/   (ROOT = this directory)
  Git repo:    deploy/zerogpu/app.py                                                    (ROOT = repository root)

Run locally:  python deploy/zerogpu/app.py   (PDEFLOW_WEB_DIR=<export dir>, PORT=7860)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "backend" / "pdeflow").is_dir() else HERE.parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

# Host-appropriate defaults; explicit environment variables still win (docs/ZEROGPU.md §4).
os.environ.setdefault("PDEFLOW_NUM_THREADS", str(min(4, os.cpu_count() or 2)))
os.environ.setdefault("PDEFLOW_CORS_ORIGINS", "")  # same origin: no CORS needed

try:  # ZeroGPU: `spaces` must be imported before torch. It is a no-op outside ZeroGPU.
    import spaces
except ImportError:  # pragma: no cover - Python 3.11 dev environment without the package
    spaces = None

import hashlib
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pdeflow import __version__
from pdeflow.api.app import create_app, validate_permeability
from pdeflow.api.schemas import ActivityItem
from pdeflow.api.settings import Settings
from pdeflow.evaluation.metrics import PRIMARY, SECONDARY, relative_l2
from pdeflow.logging_utils import get_logger, log_event

STRATEGY_CHOICES = ["auto", "fno_native", "fno64_interpolated", "fno128_native"]
ATTRIBUTION_MD = (
    "Data: **PDEBench** 2D Darcy Flow β = 1.0 (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks), DaRUS "
    "[doi:10.18419/darus-2986](https://doi.org/10.18419/darus-2986), **CC BY 4.0**. The 24 bundled samples are from "
    "this project's held-out test split. PDEFlow code and checkpoint: MIT. The data stays CC BY 4.0, and third-party "
    "components keep their own licences (NOTICE.md)."
)


# ------------------------------------------------------------------ inference used by the Gradio UI and its API
def _prediction_sha256(u: np.ndarray) -> str:
    """Hash of the float64 prediction exactly as /api/predict serialises it (C-order float64)."""
    return hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).tobytes()).hexdigest()


def _record(api: FastAPI, kind: str, sample_id: str | None, pred, warnings: list[str], rel_off: float | None) -> None:
    """Same activity feed and JSON log events as /api/predict, tagged with request_id gradio-<hex>."""
    st = api.state
    st.activity.appendleft(ActivityItem(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"), input_kind=kind, sample_id=sample_id,
        resolution=pred.resolution, strategy=pred.strategy, inference_ms=round(pred.inference_ms, 3),
        n_warnings=len(warnings), rel_l2_vs_official=rel_off))
    st.activity_total += 1
    log_event(get_logger("pdeflow.api"), "prediction", request_id=f"gradio-{uuid.uuid4().hex}", sample_id=sample_id,
              resolution=pred.resolution, strategy=pred.strategy, inference_ms=round(pred.inference_ms, 3),
              warnings=len(warnings), rel_l2_official=rel_off, channel="gradio")


def _strategy(s: str | None) -> str | None:
    return None if s in (None, "", "auto") else s


def predict_benchmark(api: FastAPI, sample_id: str, resolution: int, strategy: str | None) -> dict:
    """Bundled test sample. Identical code path to POST /api/predict {sample_id, resolution, strategy}."""
    st = api.state
    k = st.samples.position(sample_id)  # HTTPException(404) for unknown ids
    a, off, ref = st.samples.fields(k, int(resolution))
    pred = st.predictor.predict(a, _strategy(strategy))
    u = pred.u
    ev = {PRIMARY: float(relative_l2(u[None], off[None])[0]),
          SECONDARY: float(relative_l2(u[None], ref[None])[0]),
          "official_target_vs_reference": float(relative_l2(off[None], ref[None])[0])}
    _record(api, "benchmark_example", sample_id, pred, [], ev[PRIMARY])
    return {"input_kind": "benchmark_example", "sample_id": sample_id, "resolution": pred.resolution,
            "model_resolution": pred.model_resolution, "strategy": pred.strategy, "inference_ms": pred.inference_ms,
            "checkpoint_sha256": st.predictor.sha256, "warnings": [], "evaluation": ev,
            "prediction_sha256": _prediction_sha256(u), "permeability": a, "prediction": u,
            "official_pdebench_target": off, "independent_steady_state_reference": ref}


def predict_custom(api: FastAPI, rows: list[list[float]], strategy: str | None) -> dict:
    """User field. Identical code path to POST /api/predict {permeability, strategy}: same validation, same warnings.
    No ground truth exists, so no target and no error metric are produced."""
    st = api.state
    a, warnings = validate_permeability(rows)  # HTTPException(422) for invalid fields
    pred = st.predictor.predict(a, _strategy(strategy))
    _record(api, "custom", None, pred, warnings, None)
    return {"input_kind": "custom", "sample_id": None, "resolution": pred.resolution,
            "model_resolution": pred.model_resolution, "strategy": pred.strategy, "inference_ms": pred.inference_ms,
            "checkpoint_sha256": st.predictor.sha256, "warnings": warnings,
            "prediction_sha256": _prediction_sha256(pred.u), "permeability": a, "prediction": pred.u}


def parse_field_text(text: str) -> list[list[float]]:
    """JSON 2-D array, or CSV/whitespace rows (row i = x index i; '#' comments), as the web frontend accepts.
    Shape and value checks are left to the API's validate_permeability."""
    t = (text or "").strip()
    if not t:
        raise ValueError("no data provided")
    if t[0] in "[{":
        try:
            v = json.loads(t)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e.msg}") from None
        if not (isinstance(v, list) and v and all(isinstance(r, list) for r in v)):
            raise ValueError("expected a 2-D array of numbers")
        rows = v
    else:
        rows = [re.split(r"[\s,;]+", ln.strip()) for ln in t.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    try:
        return [[float(x) for x in r if x != ""] for r in rows]
    except (TypeError, ValueError):
        raise ValueError("the field contains non-numeric values") from None


def public_summary(r: dict) -> dict:
    """JSON-safe result without the large arrays (the arrays are summarised by prediction_sha256)."""
    return {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}


# ------------------------------------------------------------------ figures
def _panel(ax, z, title, cmap, vmin=None, vmax=None, label=""):
    im = ax.imshow(np.asarray(z).T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.colorbar(im, ax=ax, shrink=0.85, label=label)


def figure_benchmark(r: dict):
    u, off = r["prediction"], r["official_pdebench_target"]
    lo, hi = float(min(u.min(), off.min())), float(max(u.max(), off.max()))
    err = np.abs(u - off)
    fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    _panel(axs[0, 0], r["permeability"], "Permeability a(x, y)", "cividis", label="a")
    _panel(axs[0, 1], u, "Predicted pressure û", "viridis", lo, hi, "û")
    _panel(axs[1, 0], off, "Official PDEBench target u", "viridis", lo, hi, "u")
    _panel(axs[1, 1], err, "Absolute error |û − u|", "inferno", 0, float(err.max()), "|û − u|")
    return fig


def figure_reference(r: dict):
    u, ref = r["prediction"], r["independent_steady_state_reference"]
    err = np.abs(u - ref)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    _panel(axs[0], ref, "Steady-state reference u_ref", "viridis", label="u_ref")
    _panel(axs[1], err, "Absolute error |û − u_ref|", "inferno", 0, float(err.max()), "|û − u_ref|")
    return fig


def figure_custom(r: dict):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    _panel(axs[0], r["permeability"], "Permeability a(x, y)", "cividis", label="a")
    _panel(axs[1], r["prediction"], "Predicted pressure û", "viridis", label="û")
    return fig


def _md_benchmark(r: dict) -> str:
    ev = r["evaluation"]
    return (f"**Benchmark example** `{r['sample_id']}` at {r['resolution']}×{r['resolution']} "
            f"(`{r['strategy']}`, model run at {r['model_resolution']}×{r['model_resolution']}).\n\n"
            f"**Relative L2 vs official PDEBench target: {ev[PRIMARY] * 100:.2f}%** (primary benchmark metric)\n\n"
            f"Inference {r['inference_ms']:.1f} ms (CPU) · checkpoint `{r['checkpoint_sha256'][:12]}`")


def _md_reference(r: dict) -> str:
    ev = r["evaluation"]
    return ("**Secondary analysis** (independent steady-state reference computed by this project; not used for "
            "training, model selection or the benchmark number):\n\n"
            f"- Relative L2 vs steady-state reference: {ev[SECONDARY] * 100:.2f}%\n"
            f"- Official target vs reference: {ev['official_target_vs_reference'] * 100:.2f}%")


def _md_custom(r: dict) -> str:
    w = "".join(f"\n- ⚠️ {x}" for x in r["warnings"])
    ood = ("\n\n**Input outside the training distribution.** The model was trained only on binary fields with "
           "values 0.1 and 1.0, so this prediction is an extrapolation." + w) if r["warnings"] else ""
    return (f"**Custom input**, {r['resolution']}×{r['resolution']} (`{r['strategy']}`). Inference "
            f"{r['inference_ms']:.1f} ms · checkpoint `{r['checkpoint_sha256'][:12]}`.\n\n"
            "No ground truth exists for custom inputs, so no target, error map or error metric is shown." + ood)


# ------------------------------------------------------------------ Gradio UI
def _gr_error(e: Exception) -> gr.Error:
    if isinstance(e, HTTPException):
        return gr.Error(str(e.detail))
    return gr.Error(str(e))


def build_gradio(api: FastAPI, settings: Settings) -> gr.Blocks:
    with np.load(settings.samples_file, allow_pickle=False) as z:
        sample_ids = [f"test-{int(i)}" for i in z["indices"]]

    def ui_benchmark(sample_id, resolution, strategy):
        try:
            r = predict_benchmark(api, sample_id, int(resolution), strategy)
        except (HTTPException, ValueError) as e:
            raise _gr_error(e) from None
        figs = (figure_benchmark(r), figure_reference(r))
        out = (figs[0], _md_benchmark(r), figs[1], _md_reference(r), public_summary(r))
        plt.close("all")
        return out

    def ui_custom(file, text, strategy):
        try:
            raw = Path(file).read_text() if file else text
            r = predict_custom(api, parse_field_text(raw), strategy)
        except (HTTPException, ValueError, OSError) as e:
            raise _gr_error(e) from None
        fig = figure_custom(r)
        out = (fig, _md_custom(r), public_summary(r))
        plt.close("all")
        return out

    with gr.Blocks(title="PDEFlow · Gradio") as demo:
        gr.Markdown("# PDEFlow: FNO surrogate for 2D Darcy flow\n"
                    "−∇·(a∇u) = 1 on (0,1)², u = 0 on the boundary. Fourier Neural Operator trained on PDEBench. "
                    "[Full web app](/) · [REST API docs](/api/docs)")
        with gr.Tab("Benchmark example"):
            with gr.Row():
                sid = gr.Dropdown(sample_ids, value="test-4701" if "test-4701" in sample_ids else sample_ids[0],
                                  label="Held-out test sample")
                res = gr.Radio([64, 128], value=128, label="Resolution")
                strat = gr.Radio(STRATEGY_CHOICES, value="auto", label="Inference path (auto: 64 native / 128 interpolated)")
            run = gr.Button("Run prediction", variant="primary")
            fig_main = gr.Plot(label="Comparison with the official PDEBench target")
            md_main = gr.Markdown()
            with gr.Accordion("Secondary analysis: independent steady-state reference", open=False):
                fig_ref = gr.Plot(label="Steady-state reference")
                md_ref = gr.Markdown()
            js_b = gr.JSON(label="Result summary (same numbers as /api/predict)")
            run.click(ui_benchmark, [sid, res, strat], [fig_main, md_main, fig_ref, md_ref, js_b],
                      api_name="predict_benchmark")
        with gr.Tab("Custom field"):
            gr.Markdown("Upload or paste a 64×64 or 128×128 permeability grid (JSON array or CSV rows, a > 0). "
                        "Training data are binary {0.1, 1.0}; other values are flagged as out of distribution.")
            with gr.Row():
                f_in = gr.File(label="Field file (.json / .csv / .txt)", file_types=[".json", ".csv", ".txt"], type="filepath")
                t_in = gr.Textbox(label="…or paste here", lines=6)
            strat_c = gr.Radio(STRATEGY_CHOICES, value="auto", label="Inference path")
            run_c = gr.Button("Run prediction", variant="primary")
            fig_c = gr.Plot(label="Prediction")
            md_c = gr.Markdown()
            js_c = gr.JSON(label="Result summary (same numbers as /api/predict)")
            run_c.click(ui_custom, [f_in, t_in, strat_c], [fig_c, md_c, js_c], api_name="predict_custom")
        # pinned SHA; the loaded checkpoint is verified against it at startup (the app refuses to start otherwise)
        gr.Markdown(ATTRIBUTION_MD + f"\n\nService v{__version__} · checkpoint SHA-256 `{settings.expected_sha256}`")
    return demo


# ------------------------------------------------------------------ assembly
def _web_dir() -> Path | None:
    for cand in (os.environ.get("PDEFLOW_WEB_DIR"), ROOT / "web", HERE / "web"):
        if cand and (Path(cand) / "index.html").is_file():
            return Path(cand).resolve()
    return None


def build_app(settings: Settings | None = None, web_dir: Path | None = None) -> FastAPI:
    settings = settings or Settings()
    api = create_app(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # The API's own startup: checkpoint SHA-256 verification, model + samples load. Fails fast on mismatch.
        async with api.router.lifespan_context(api):
            yield

    root = FastAPI(title="PDEFlow (ZeroGPU Space)", version=__version__, lifespan=lifespan, docs_url=None,
                   redoc_url=None, openapi_url=None)
    root.state.api = api
    root.mount("/api", api)
    root = gr.mount_gradio_app(root, build_gradio(api, settings), path="/gradio")

    web = web_dir if web_dir is not None else _web_dir()
    log = get_logger("pdeflow.space")
    log_event(log, "space_app_built", web_dir=str(web) if web else None, root=str(ROOT))
    if web is not None:
        if (web / "_next").is_dir():
            root.mount("/_next", StaticFiles(directory=web / "_next"), name="next-assets")

        @root.api_route("/{page:path}", methods=["GET", "HEAD"], include_in_schema=False)
        def static_page(page: str, request: Request) -> Response:
            # Clean URLs of the static export: / -> index.html, /results -> results.html, assets as files.
            # GET and HEAD: the Next.js client prefetches pages with HEAD.
            for cand in (web / page, web / f"{page}.html", web / page / "index.html"):
                try:
                    c = cand.resolve()
                except OSError:
                    continue
                if c.is_file() and web in c.parents:
                    return FileResponse(c)
            return FileResponse(web / "404.html", status_code=404) if (web / "404.html").is_file() else \
                Response(status_code=404)
    else:
        @root.get("/", include_in_schema=False)
        def no_web() -> RedirectResponse:  # web export not built: the Gradio UI is the front page
            return RedirectResponse("/gradio/")
    return root


def main() -> None:
    import uvicorn

    t0 = time.perf_counter()
    app = build_app()
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    log_event(get_logger("pdeflow.space"), "space_starting", host=host, port=port,
              zero_gpu=os.environ.get("SPACES_ZERO_GPU") == "true", build_s=round(time.perf_counter() - t0, 2))
    uvicorn.run(app, host=host, port=port, access_log=False, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
