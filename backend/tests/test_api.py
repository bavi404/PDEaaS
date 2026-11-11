"""FastAPI service: checkpoint identity, endpoint contracts, validation, target separation, logging."""

import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # starlette's httpx deprecation notice
    from fastapi.testclient import TestClient

from pdeflow.api.app import create_app
from pdeflow.api.settings import EXPECTED_CHECKPOINT_SHA256, Settings
from pdeflow.evaluation.metrics import PRIMARY, SECONDARY, relative_l2
from pdeflow.inference import ChecksumMismatch, Predictor
from pdeflow.training.core import sha256_file

ROOT = Path(__file__).resolve().parents[2]
REPORT = json.loads((ROOT / "reports/m2_eval_fno_baseline.json").read_text())
BIN64 = np.where(np.add.outer(np.arange(64), np.arange(64)) % 7 < 3, 1.0, 0.1)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings())) as c:
        yield c


@pytest.fixture(scope="module")
def bundle():
    with np.load(Settings().samples_file) as z:
        return {k: z[k] for k in z.files}


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.getMessage(), getattr(record, "fields", {})))


# ------------------------------------------------------------------ checkpoint identity
def test_served_checkpoint_is_the_evaluated_checkpoint(client):
    h = client.get("/health").json()
    assert h == {"status": "ok", "model_loaded": True, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "n_samples": 24}
    assert sha256_file(Settings().checkpoint) == EXPECTED_CHECKPOINT_SHA256 == REPORT["checkpoint_sha256"]
    summary = json.loads((ROOT / "reports/m2_run_fno_baseline/summary.json").read_text())
    assert summary["best_checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256


def test_startup_refuses_wrong_checkpoint_hash():
    with pytest.raises(ChecksumMismatch), TestClient(create_app(replace(Settings(), expected_sha256="0" * 64))):
        pass


def test_model_card_metrics_come_from_matching_report(client):
    m = client.get("/model").json()
    assert m["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256 and m["checkpoint"]["n_params"] == 357_217
    assert m["checkpoint"]["epoch"] == 139
    tm = m["test_metrics"]
    assert tm["res64_fno_native"][PRIMARY] == REPORT["res64"][PRIMARY]["relative_l2"]
    assert tm["res64_fno_native"][SECONDARY] == REPORT["res64"][SECONDARY]["relative_l2"]
    assert tm["res128_fno128_native"][PRIMARY] == REPORT["res128"][PRIMARY]["relative_l2"]
    assert tm["res128_fno64_interpolated"][PRIMARY] == REPORT["res128_interpolation_baseline"][PRIMARY]["relative_l2"]
    assert "CC BY 4.0" in m["dataset"]


def test_model_card_omits_metrics_from_a_different_checkpoint(tmp_path):
    bad = dict(REPORT, checkpoint_sha256="f" * 64)
    (tmp_path / "r.json").write_text(json.dumps(bad))
    with TestClient(create_app(replace(Settings(), eval_report=tmp_path / "r.json"))) as c:
        assert c.get("/model").json()["test_metrics"] is None


# ------------------------------------------------------------------ samples
def test_samples_list(client, bundle):
    s = client.get("/samples").json()
    assert len(s) == 24 and len({x["id"] for x in s}) == 24
    assert {x["test_index"] for x in s} == set(bundle["indices"].tolist())
    assert {2113, 4701} <= {x["test_index"] for x in s}


@pytest.mark.parametrize("res", [64, 128])
def test_sample_detail_keeps_targets_separate(client, bundle, res):
    d = client.get(f"/samples/test-4701?resolution={res}").json()
    k = int(np.flatnonzero(bundle["indices"] == 4701)[0])
    s = 128 // res
    assert np.array(d["permeability"]).shape == (res, res)
    assert np.array_equal(np.array(d["targets"]["official_pdebench_target"]), bundle["u_official"][k, ::s, ::s])
    assert np.array_equal(np.array(d["targets"]["independent_steady_state_reference"]), bundle["u_reference"][k, ::s, ::s])
    assert not np.array_equal(bundle["u_official"][k], bundle["u_reference"][k])


def test_sample_errors(client):
    assert client.get("/samples/test-0").status_code == 404
    assert client.get("/samples/test-4701?resolution=96").status_code == 422


# ------------------------------------------------------------------ predict: correctness
def test_predict_sample_reproduces_evaluated_worst_case(client):
    """Test idx 4701 is the worst res128 (native) test sample in the evaluation report.

    Tolerance: the report ran this sample inside a batch of 64; the API runs it alone. float32 CPU kernels round
    batch-size-dependently (measured max |du| 3.0e-7 at max |u| 0.89, rel-L2 changes by 9e-8 relative). In the
    same batch the value is bit-identical (scripts/verify_api_inference.py)."""
    r = client.post("/predict", json={"sample_id": "test-4701", "strategy": "fno128_native"}).json()
    assert r["evaluation"]["primary_vs_official_pdebench_target"]["relative_l2"] == pytest.approx(
        REPORT["res128"][PRIMARY]["relative_l2"]["max"], rel=1e-6)


@pytest.mark.parametrize("res,strategy", [(64, "fno_native"), (128, "fno64_interpolated"), (128, "fno128_native")])
def test_predict_sample_matches_predictor_and_metrics(client, bundle, res, strategy):
    r = client.post("/predict", json={"sample_id": "test-2113", "resolution": res, "strategy": strategy,
                                      "return_targets": True}).json()
    k = int(np.flatnonzero(bundle["indices"] == 2113)[0])
    s = 128 // res
    a = bundle["a"][k, ::s, ::s].astype(np.float64)
    off, ref = (bundle[n][k, ::s, ::s].astype(np.float64) for n in ("u_official", "u_reference"))
    direct = Predictor(Settings().checkpoint, EXPECTED_CHECKPOINT_SHA256, 4).predict(a, strategy).u
    pred = np.array(r["prediction"])
    assert pred.shape == (res, res) and r["strategy"] == strategy and r["resolution"] == res
    assert r["model_resolution"] == (64 if strategy != "fno128_native" else 128)
    np.testing.assert_array_equal(pred, direct)
    ev = r["evaluation"]
    assert ev["primary_vs_official_pdebench_target"]["relative_l2"] == pytest.approx(relative_l2(pred[None], off[None])[0], rel=1e-12)
    assert ev["secondary_vs_independent_steady_state_reference"]["relative_l2"] == pytest.approx(relative_l2(pred[None], ref[None])[0], rel=1e-12)
    assert ev["official_target_vs_reference"]["relative_l2"] == pytest.approx(relative_l2(off[None], ref[None])[0], rel=1e-12)
    assert np.array_equal(np.array(r["targets"]["official_pdebench_target"]), off)
    assert r["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256 and r["inference_ms"] > 0


def test_user_field_equals_sample_path(client):
    d = client.get("/samples/test-2113?resolution=128").json()
    by_field = client.post("/predict", json={"permeability": d["permeability"]}).json()
    by_id = client.post("/predict", json={"sample_id": "test-2113"}).json()
    assert by_field["prediction"] == by_id["prediction"] and by_field["warnings"] == []
    assert "evaluation" not in by_field and "targets" not in by_field


def test_default_strategies(client):
    assert client.post("/predict", json={"permeability": BIN64.tolist()}).json()["strategy"] == "fno_native"
    a128 = np.kron(BIN64, np.ones((2, 2)))
    r = client.post("/predict", json={"permeability": a128.tolist()}).json()
    assert r["strategy"] == "fno64_interpolated" and np.array(r["prediction"]).shape == (128, 128)


def test_predict_is_deterministic_under_concurrency(client):
    body = {"permeability": BIN64.tolist()}
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: client.post("/predict", json=body).json()["prediction"], range(6)))
    assert all(o == outs[0] for o in outs)
    assert np.isfinite(np.array(outs[0])).all() and np.array(outs[0]).min() > -0.05


# ------------------------------------------------------------------ predict: validation
@pytest.mark.parametrize("body,fragment", [
    ({"permeability": [[1.0] * 64] * 63}, "square grid"),
    ({"permeability": [[1.0] * 96] * 96}, "64x64 or 128x128"),
    ({"permeability": [[1.0] * 64] * 63 + [[1.0] * 63]}, "square grid"),
    ({"permeability": [[-1.0] * 64] * 64}, "strictly positive"),
    ({"permeability": [[0.0] * 64] * 64}, "strictly positive"),
    ({"permeability": [[1e7] * 64] * 64}, "above"),
    ({"permeability": BIN64.tolist(), "strategy": "fno128_native"}, "not valid for 64x64"),
])
def test_predict_rejects_invalid_input(client, body, fragment):
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and fragment in str(r.json()["detail"]) and r.json()["request_id"]


def test_predict_rejects_non_finite(client):
    row = ", ".join(["1.0"] * 63 + ["NaN"])
    body = '{"permeability": [' + ", ".join(f"[{row}]" for _ in range(64)) + "]}"
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "non-finite" in r.json()["detail"]


@pytest.mark.parametrize("body", [
    {}, {"permeability": BIN64.tolist(), "sample_id": "test-2113"},
    {"permeability": BIN64.tolist(), "return_targets": True},
    {"permeability": BIN64.tolist(), "unknown": 1},
    {"sample_id": "test-2113", "resolution": 96},
    {"sample_id": "test-2113", "strategy": "magic"},
])
def test_predict_schema_errors(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_sample_is_404(client):
    assert client.post("/predict", json={"sample_id": "test-0"}).status_code == 404


def test_out_of_distribution_warnings(client):
    r = client.post("/predict", json={"permeability": (BIN64 * 5).tolist()}).json()
    assert any("outside the training range" in w for w in r["warnings"])
    r = client.post("/predict", json={"permeability": [[0.5] * 64] * 64}).json()
    assert any("non-binary" in w for w in r["warnings"])
    assert client.post("/predict", json={"permeability": BIN64.tolist()}).json()["warnings"] == []


def test_body_size_limit():
    with TestClient(create_app(replace(Settings(), max_body_bytes=1000))) as c:
        r = c.post("/predict", json={"permeability": BIN64.tolist()})
        assert r.status_code == 413 and r.json()["request_id"]


# ------------------------------------------------------------------ logging / request ids
def test_request_id_and_structured_logs(client):
    cap = _Capture()
    logger = logging.getLogger("pdeflow.api")
    logger.addHandler(cap)
    try:
        r = client.post("/predict", json={"sample_id": "test-2113"}, headers={"X-Request-ID": "abc-123"})
        bad = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    finally:
        logger.removeHandler(cap)
    assert r.headers["X-Request-ID"] == "abc-123"
    assert bad.headers["X-Request-ID"] != "bad id with spaces" and len(bad.headers["X-Request-ID"]) == 32
    events = {(m, f.get("request_id")) for m, f in cap.records}
    assert ("prediction", "abc-123") in events and ("request", "abc-123") in events
    req = next(f for m, f in cap.records if m == "request" and f["request_id"] == "abc-123")
    assert req["status"] == 200 and req["path"] == "/predict" and req["duration_ms"] > 0
    pred = next(f for m, f in cap.records if m == "prediction")
    assert pred["strategy"] == "fno64_interpolated" and pred["rel_l2_official"] > 0


def test_cors_allows_configured_origin_only(client):
    ok = client.options("/predict", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    bad = client.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


# ------------------------------------------------------------------ bundled data integrity
@pytest.mark.data
def test_bundle_matches_processed_test_split(bundle):
    t = np.load(ROOT / "data/processed/darcy_beta1.0_test.npz")
    r = np.load(ROOT / "data/processed/darcy_beta1.0_test_reference.npz")
    pos = [int(np.flatnonzero(t["indices"] == i)[0]) for i in bundle["indices"]]
    assert np.array_equal(bundle["a"], t["a"][pos])
    assert np.array_equal(bundle["u_official"], t["u"][pos])
    assert np.array_equal(bundle["u_reference"], r["u_reference"][pos])
    splits = json.loads((ROOT / "data/splits/darcy_beta1.0_seed20261111.json").read_text())
    assert set(bundle["indices"].tolist()) <= set(splits["test"])


def test_inference_equivalence_report_is_for_served_checkpoint():
    """scripts/verify_api_inference.py: whole test split through Predictor == evaluation report (rtol 1e-9)."""
    eq = json.loads((ROOT / "reports/m4_inference_equivalence.json").read_text())
    assert eq["all_match"] is True and eq["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert set(eq["cases"]) == {"res64", "res128", "res128_interpolation_baseline"}
    assert all(c["max_abs_rel_diff"] <= 1e-9 for c in eq["cases"].values())


# ------------------------------------------------------------------ activity / input kind
def test_input_kind_and_activity_feed():
    with TestClient(create_app(Settings())) as c:
        assert c.get("/activity").json()["items"] == []
        r1 = c.post("/predict", json={"sample_id": "test-2113", "resolution": 64}).json()
        r2 = c.post("/predict", json={"permeability": BIN64.tolist()}).json()
        c.post("/predict", json={"permeability": [[1.0] * 10] * 10})  # rejected: not recorded
        assert r1["input_kind"] == "benchmark_example" and r2["input_kind"] == "custom"
        act = c.get("/activity", params={"limit": 5}).json()
        assert act["total_since_start"] == 2 and len(act["items"]) == 2
        newest, oldest = act["items"]
        assert newest["input_kind"] == "custom" and newest["rel_l2_vs_official"] is None and newest["sample_id"] is None
        assert oldest["input_kind"] == "benchmark_example" and oldest["sample_id"] == "test-2113"
        assert oldest["rel_l2_vs_official"] == r1["evaluation"]["primary_vs_official_pdebench_target"]["relative_l2"]
        assert set(newest) == {"timestamp_utc", "input_kind", "sample_id", "resolution", "strategy", "inference_ms",
                               "n_warnings", "rel_l2_vs_official"}  # metadata only, no fields
        assert c.get("/activity", params={"limit": 0}).status_code == 422
