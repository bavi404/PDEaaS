"""ZeroGPU Space app: routes, checkpoint identity, and Gradio <-> FastAPI inference equivalence.

Equivalence is exact: the Gradio path must return the same prediction (SHA-256 of the float64 array as /api/predict
serialises it) and the same metrics as /api/predict for identical inputs.
"""

import hashlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import replace

import numpy as np
import pytest
from conftest import APP_PATH, REPO
from pdeflow.api.settings import EXPECTED_CHECKPOINT_SHA256, Settings
from pdeflow.evaluation.metrics import PRIMARY, SECONDARY
from pdeflow.training.core import sha256_file

REPORT = json.loads((REPO / "reports/m2_eval_fno_baseline.json").read_text())
BIN64 = np.where(np.add.outer(np.arange(64), np.arange(64)) % 7 < 3, 1.0, 0.1)


def sha_of(grid) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(grid, dtype=np.float64)).tobytes()).hexdigest()


# ------------------------------------------------------------------ checkpoint identity / startup
def test_checkpoint_sha_unchanged_and_served(client):
    assert sha256_file(Settings().checkpoint) == EXPECTED_CHECKPOINT_SHA256 == REPORT["checkpoint_sha256"]
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256 and h["n_samples"] == 24


def test_startup_refuses_wrong_checkpoint_hash(space, web_dir):
    from fastapi.testclient import TestClient
    from pdeflow.inference import ChecksumMismatch

    with pytest.raises(ChecksumMismatch), TestClient(space.build_app(replace(Settings(), expected_sha256="0" * 64),
                                                                     web_dir=web_dir)):
        pass


# ------------------------------------------------------------------ /api is the unchanged FastAPI app
def test_api_routes_identical_to_standalone_api(client):
    from fastapi.testclient import TestClient
    from pdeflow.api.app import create_app

    with TestClient(create_app(Settings())) as ref:
        assert client.get("/api/samples").json() == ref.get("/samples").json()
        m, rm = client.get("/api/model").json(), ref.get("/model").json()
        assert m["test_metrics"] == rm["test_metrics"] and m["dataset"] == rm["dataset"]
        assert {k: v for k, v in m["checkpoint"].items() if k != "num_threads"} == \
               {k: v for k, v in rm["checkpoint"].items() if k != "num_threads"}
        for body in ({"sample_id": "test-2113", "resolution": 64}, {"sample_id": "test-4701", "resolution": 128},
                     {"permeability": BIN64.tolist()}):
            a, b = client.post("/api/predict", json=body).json(), ref.post("/predict", json=body).json()
            assert a["prediction"] == b["prediction"] and a.get("evaluation") == b.get("evaluation")
        assert client.post("/api/predict", json={"permeability": [[1, 2], [3, 4]]}).status_code == 422
        assert client.get("/api/samples/test-0").status_code == 404
        assert client.get("/api/activity").status_code == 200
        r = client.get("/api/health", headers={"X-Request-ID": "space-rid-1"})
        assert r.headers["X-Request-ID"] == "space-rid-1"


def test_reported_worst_case_through_space(client):
    r = client.post("/api/predict", json={"sample_id": "test-4701", "resolution": 128, "strategy": "fno128_native"}).json()
    assert r["evaluation"][PRIMARY]["relative_l2"] == pytest.approx(REPORT["res128"][PRIMARY]["relative_l2"]["max"], rel=1e-6)


# ------------------------------------------------------------------ static frontend
@pytest.mark.parametrize("path", ["/", "/workspace", "/results", "/model"])
def test_static_pages_served_with_clean_urls(client, path):
    for method in ("get", "head"):  # the Next.js client prefetches with HEAD
        r = getattr(client, method)(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), (method, path)
    assert "PDEFlow" in client.get(path).text


def test_static_assets_and_404(client, web_dir):
    js = next((web_dir / "_next/static").rglob("*.js")).relative_to(web_dir).as_posix()
    assert client.get("/" + js).status_code == 200
    assert client.get("/favicon.svg").status_code == 200
    assert client.get("/build-info.json").json()["mode"].startswith("static export")
    assert client.get("/does-not-exist").status_code == 404
    for evil in ("/..%2f..%2fbackend%2fmodels%2ffno_darcy_beta1_baseline.pt", "/../../etc/passwd", "/%2e%2e/LICENSE"):
        assert client.get(evil).status_code == 404, evil


def test_static_export_has_no_proxy_route(web_dir):
    """The frontend calls same-origin /api; the Next.js proxy route must not be in the export."""
    assert not (web_dir / "api").exists()
    info = json.loads((web_dir / "build-info.json").read_text())
    assert info["next"] == "16.4.3"


def test_api_not_shadowed_by_static_routes(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/docs").status_code == 200


def test_without_web_export_root_redirects_to_gradio(space, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(space, "_web_dir", lambda: None)  # simulate a Space without the static export
    with TestClient(space.build_app(web_dir=None)) as c:
        r = c.get("/", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == "/gradio/"
        assert c.get("/api/health").status_code == 200 and c.get("/gradio/").status_code == 200


# ------------------------------------------------------------------ Gradio
def test_gradio_page_and_attribution(client):
    r = client.get("/gradio/")
    assert r.status_code == 200
    cfg = client.get("/gradio/config").json()
    text = json.dumps(cfg)
    for needle in ("CC BY 4.0", "doi:10.18419/darus-2986", "PDEBench", EXPECTED_CHECKPOINT_SHA256, "MIT"):
        assert needle in text, needle


@pytest.mark.parametrize("sample_id,res,strategy", [("test-2113", 64, "auto"), ("test-2113", 128, "auto"),
                                                    ("test-4701", 128, "fno128_native"), ("test-8142", 128, "fno64_interpolated")])
def test_gradio_benchmark_matches_api(space, client, sample_id, res, strategy):
    api = client.app.state.api
    g = space.predict_benchmark(api, sample_id, res, strategy)
    body = {"sample_id": sample_id, "resolution": res, "return_targets": True}
    if strategy != "auto":
        body["strategy"] = strategy
    a = client.post("/api/predict", json=body).json()
    assert g["prediction_sha256"] == sha_of(a["prediction"])
    assert g["strategy"] == a["strategy"] and g["resolution"] == a["resolution"] and g["model_resolution"] == a["model_resolution"]
    for k in (PRIMARY, SECONDARY, "official_target_vs_reference"):
        assert g["evaluation"][k] == a["evaluation"][k]["relative_l2"]
    # targets kept separate and identical to the API's
    assert sha_of(g["official_pdebench_target"]) == sha_of(a["targets"]["official_pdebench_target"])
    assert sha_of(g["independent_steady_state_reference"]) == sha_of(a["targets"]["independent_steady_state_reference"])
    assert g["checkpoint_sha256"] == a["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256


@pytest.mark.parametrize("field,strategy", [(BIN64, "auto"), (np.kron(BIN64, np.ones((2, 2))), "auto"),
                                            (np.kron(BIN64, np.ones((2, 2))), "fno128_native"), (BIN64 * 3, "auto")])
def test_gradio_custom_matches_api(space, client, field, strategy):
    api = client.app.state.api
    g = space.predict_custom(api, field.tolist(), strategy)
    body = {"permeability": field.tolist()} | ({} if strategy == "auto" else {"strategy": strategy})
    a = client.post("/api/predict", json=body).json()
    assert g["prediction_sha256"] == sha_of(a["prediction"]) and g["warnings"] == a["warnings"]
    assert "evaluation" not in g and "official_pdebench_target" not in g  # no fabricated ground truth
    assert a["input_kind"] == g["input_kind"] == "custom"


def test_gradio_custom_validation_matches_api(space, client):
    from fastapi import HTTPException

    api = client.app.state.api
    for bad in ([[1.0, 2.0], [3.0, 4.0]], [[-1.0] * 64] * 64, [[float("nan")] * 64] * 64):
        with pytest.raises(HTTPException) as e:
            space.predict_custom(api, bad, "auto")
        assert e.value.status_code == 422
    with pytest.raises(ValueError):
        space.predict_benchmark(api, "test-2113", 64, "fno128_native")


@pytest.mark.parametrize("text,ok", [("[[0.1, 1.0], [1.0, 0.1]]", True), ("# c\n0.1,1.0\n1.0 0.1", True),
                                     ('{"a": 1}', False), ("[[1, 2], [3", False), ("0.1,abc", False), ("", False)])
def test_parse_field_text(space, text, ok):
    if ok:
        assert space.parse_field_text(text) == [[0.1, 1.0], [1.0, 0.1]]
    else:
        with pytest.raises(ValueError):
            space.parse_field_text(text)


def test_gradio_predictions_recorded_in_activity(space, client):
    api = client.app.state.api
    before = client.get("/api/activity").json()["total_since_start"]
    space.predict_benchmark(api, "test-2113", 64, "auto")
    act = client.get("/api/activity").json()
    assert act["total_since_start"] == before + 1 and act["items"][0]["sample_id"] == "test-2113"


def test_gradio_http_api_matches_fastapi(live_url, tmp_path):
    """End-to-end over HTTP: gradio_client -> /gradio vs plain POST /api/predict on the same server."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gradio_client import Client, handle_file
    import httpx

    c = Client(f"{live_url}/gradio/", verbose=False)
    for args, body in ((("test-4701", 128, "fno128_native"), {"sample_id": "test-4701", "resolution": 128, "strategy": "fno128_native"}),
                       (("test-2113", 64, "auto"), {"sample_id": "test-2113", "resolution": 64})):
        out = c.predict(*args, api_name="/predict_benchmark")
        summary = out[-1]
        a = httpx.post(f"{live_url}/api/predict", json=body, timeout=60).json()
        assert summary["prediction_sha256"] == sha_of(a["prediction"])
        assert summary["evaluation"][PRIMARY] == a["evaluation"][PRIMARY]["relative_l2"]
        assert "Relative L2 vs official PDEBench target" in out[1] and "Secondary analysis" in out[3]
    f = tmp_path / "field.json"
    f.write_text(json.dumps((BIN64 * 0.5).tolist()))
    out = c.predict(handle_file(str(f)), "", "auto", api_name="/predict_custom")
    a = httpx.post(f"{live_url}/api/predict", json={"permeability": (BIN64 * 0.5).tolist()}, timeout=60).json()
    assert out[-1]["prediction_sha256"] == sha_of(a["prediction"]) and out[-1]["warnings"] == a["warnings"]
    assert "No ground truth exists" in out[1]
    with pytest.raises(Exception, match="64x64 or 128x128"):
        c.predict(None, "[[1, 2], [3, 4]]", "auto", api_name="/predict_custom")


def test_emulated_zerogpu_environment_is_bit_identical(tmp_path):
    """Run the app under SPACES_ZERO_GPU=true (spaces patches torch) in a subprocess: predictions must be unchanged."""
    script = tmp_path / "zg.py"
    script.write_text(f"""
import importlib.util, json, sys, warnings
warnings.simplefilter("ignore")
spec = importlib.util.spec_from_file_location("zg_app", {str(APP_PATH)!r}); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from fastapi.testclient import TestClient
import spaces, torch
with TestClient(m.build_app(web_dir=None)) as c:
    api = c.app.state.api
    out = {{"zero_gpu": spaces.config.Config.zero_gpu, "cuda_emulated": torch.cuda.is_available(),
           "bench": m.predict_benchmark(api, "test-4701", 128, "fno128_native")["prediction_sha256"],
           "custom": m.predict_custom(api, [[1.0 if (i + j) % 7 < 3 else 0.1 for j in range(64)] for i in range(64)], "auto")["prediction_sha256"]}}
print("RESULT" + json.dumps(out))
""")
    def run(env_extra):
        env = {**os.environ, **env_extra}
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=600, check=False)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")), None)
        assert line, p.stderr[-2000:]
        return json.loads(line[6:])
    normal = run({"SPACES_ZERO_GPU": ""})
    zg = run({"SPACES_ZERO_GPU": "true", "SPACES_ZERO_DEVICE_API_URL": "http://127.0.0.1:9"})
    assert normal["zero_gpu"] is False and zg["zero_gpu"] is True and zg["cuda_emulated"] is True
    assert zg["bench"] == normal["bench"] and zg["custom"] == normal["custom"]
