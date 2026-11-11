"""THROWAWAY feasibility prototype (not part of the repo): one process = FastAPI root
   /api/*  -> the existing, unmodified pdeflow.api.app.create_app() (pinned-SHA checkpoint, same Predictor)
   /       -> a minimal Gradio UI calling the same Predictor object (CPU; no @spaces.GPU)."""
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, "/home/user/pdeflow/backend")
import gradio as gr
import spaces  # noqa: F401  (effect-free outside ZeroGPU, per HF docs)
import uvicorn
from fastapi import FastAPI

from pdeflow.api.app import create_app
from pdeflow.evaluation.metrics import relative_l2

api = create_app()


@asynccontextmanager
async def lifespan(_app):
    async with api.router.lifespan_context(api):  # runs the API's own startup: SHA check + model load
        yield


root = FastAPI(lifespan=lifespan)
root.mount("/api", api)


def solve(sample_id: str, resolution: int):
    st = api.state
    k = st.samples.position(sample_id)
    a, off, ref = st.samples.fields(k, int(resolution))
    pred = st.predictor.predict(a)
    return {"strategy": pred.strategy, "rel_l2_official": float(relative_l2(pred.u[None], off[None])[0]),
            "rel_l2_reference": float(relative_l2(pred.u[None], ref[None])[0]), "inference_ms": pred.inference_ms}


with gr.Blocks(title="PDEFlow (prototype)") as demo:
    sid = gr.Textbox(value="test-4701", label="sample id")
    res = gr.Radio([64, 128], value=128, label="resolution")
    out = gr.JSON()
    gr.Button("Solve").click(solve, [sid, res], out, api_name="solve")

root = gr.mount_gradio_app(root, demo, path="/gradio")
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
OUT = Path("<scratch>/fe_export/out")
root.mount("/_next", StaticFiles(directory=OUT / "_next"), name="next-assets")

@root.api_route("/{page:path}", methods=["GET", "HEAD"], include_in_schema=False)
def web(page: str):
    # clean URLs of the static export: /results -> results.html, / -> index.html, assets as files
    for cand in (OUT / page, OUT / f"{page}.html", OUT / page / "index.html"):
        if cand.is_file() and OUT in cand.resolve().parents:
            return FileResponse(cand)
    return FileResponse(OUT / "404.html", status_code=404)

if __name__ == "__main__":
    uvicorn.run(root, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 7870)
