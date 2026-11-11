"""docs/RESULTS.md numbers must match reports/m2_eval_fno_baseline.json and the committed checkpoint."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/m2_eval_fno_baseline.json"
SUMMARY = ROOT / "reports/m2_run_fno_baseline/summary.json"
DOC = ROOT / "docs/RESULTS.md"
CKPT = ROOT / "backend/models/fno_darcy_beta1_baseline.pt"
P, S = "primary_vs_official_pdebench_target", "secondary_vs_independent_steady_state_reference"

pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="M2 report not generated")


def _f4(x):
    return f"{x:.4f}"


def test_headline_numbers_in_doc():
    r, doc = json.loads(REPORT.read_text()), DOC.read_text()
    for key in ("res64", "res128", "res128_interpolation_baseline"):
        for side in (P, S):
            for stat in ("mean", "median"):
                assert _f4(r[key][side]["relative_l2"][stat]) in doc, (key, side, stat)
        for stat in ("p90", "p99", "max"):
            assert _f4(r[key][P]["relative_l2"][stat]) in doc, (key, stat)
        for reg, v in r[key]["regions"].items():
            assert _f4(v[P]["mean"]) in doc, (key, reg)
    for k, v in r["res64"][P]["strata"].items():
        assert _f4(v["mean"]) in doc and _f4(r["res64"][S]["strata"][k]["mean"]) in doc, k
    for lat in r["latency_cpu"]:
        assert f"{lat['median_ms']:,.1f}" in doc


def test_run_summary_and_checkpoint_hash():
    s, doc = json.loads(SUMMARY.read_text()), DOC.read_text()
    assert s["total_nonfinite_steps"] == 0
    assert f"{s['best_val_rel_l2_official_64']:.5f}" in doc and f"epoch {s['best_epoch']}" in doc
    assert f"{s['n_params']:,}" in doc and f"{s['train_time_s']:,.0f} s" in doc
    assert s["best_checkpoint_sha256"] in doc
    assert hashlib.sha256(CKPT.read_bytes()).hexdigest() == s["best_checkpoint_sha256"]
    assert json.loads(REPORT.read_text())["checkpoint_sha256"] == s["best_checkpoint_sha256"]
