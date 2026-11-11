"""Checks against the real PDEBench file and the committed split/normaliser artefacts."""

import json

import numpy as np
import pytest

from pdeflow.data.dataset import DarcyDataset, load_split_arrays
from pdeflow.data.normalization import DarcyNormalizers, ScalarNormalizer
from pdeflow.data.pdebench import inspect_file, read_samples
from pdeflow.data.physics import relative_residual
from pdeflow.data.resample import subsample
from pdeflow.data.splits import SplitManifest
from tests.conftest import NORMS, PROCESSED, RAW, SPLITS, require

pytestmark = pytest.mark.data
DARUS_MD5 = "81694ed31306ff2e5f6b76349b0b4389"


def test_file_layout():
    info = inspect_file(require(RAW))
    assert (info.n_samples, info.resolution, info.beta) == (10000, 128, 1.0)
    np.testing.assert_allclose(info.x, (np.arange(128) + 0.5) / 128, atol=1e-7)


def test_provenance_matches_darus():
    prov = json.loads(require(RAW.with_name(RAW.name + ".provenance.json")).read_text())
    assert prov["md5"] == DARUS_MD5 and prov["darus_file_id"] == 133219 and prov["size_bytes"] == 1310724488


def test_values_physical():
    a, u = read_samples(require(RAW), list(range(0, 10000, 97)))
    assert set(np.unique(a).tolist()) == {np.float32(0.1), np.float32(1.0)}
    assert np.isfinite(u).all() and (u > 0).all()


def test_steady_samples_satisfy_discrete_pde():
    """Highly-permeable samples are converged; their residual must be tiny under our fitted stencil."""
    a, u = read_samples(require(RAW), list(range(200)))
    frac = (a == 1.0).mean(axis=(1, 2))
    sel = frac > 0.7
    assert sel.sum() >= 5
    assert relative_residual(a[sel], u[sel], 1.0, 1 / 128).max() < 2e-3


def test_split_manifest_valid_and_matches_file():
    m = SplitManifest.load(require(SPLITS))  # .load() runs validate(): disjoint, partition, groups intact
    assert m.source_md5 == DARUS_MD5 and m.n_total == 10000
    assert (len(m.train), len(m.val), len(m.test)) == (8000, 1000, 1000)
    assert len(m.duplicate_groups) == 1 and len(m.duplicate_groups[0]) == 46


def test_processed_files_match_manifest_and_raw():
    m = SplitManifest.load(require(SPLITS))
    for split, n in (("train", 2000), ("val", 1000), ("test", 1000)):
        d = load_split_arrays(require(PROCESSED / f"darcy_beta1.0_{split}.npz"))
        assert d["indices"].tolist() == m.subset(split, n).tolist()
        a, u = read_samples(RAW, d["indices"][:5])
        np.testing.assert_array_equal(d["a"][:5], a)
        np.testing.assert_array_equal(d["u"][:5], u)
    tr = set(load_split_arrays(PROCESSED / "darcy_beta1.0_train.npz")["indices"].tolist())
    te = set(load_split_arrays(PROCESSED / "darcy_beta1.0_test.npz")["indices"].tolist())
    va = set(load_split_arrays(PROCESSED / "darcy_beta1.0_val.npz")["indices"].tolist())
    assert not (tr & te or tr & va or va & te)


def test_no_identical_inputs_across_processed_splits():
    hashes = {}
    for split in ("train", "val", "test"):
        a = load_split_arrays(require(PROCESSED / f"darcy_beta1.0_{split}.npz"))["a"]
        hashes[split] = {x.tobytes() for x in a}
    assert not (hashes["train"] & hashes["test"]) and not (hashes["train"] & hashes["val"])


def test_normalizers_fitted_on_train_only():
    norms = DarcyNormalizers.from_json(require(NORMS).read_text())
    tr = load_split_arrays(require(PROCESSED / "darcy_beta1.0_train.npz"))
    assert norms.a == ScalarNormalizer.fit(subsample(tr["a"], 2))
    assert norms.u == ScalarNormalizer.fit(subsample(tr["u"], 2))
    te = load_split_arrays(PROCESSED / "darcy_beta1.0_test.npz")
    assert norms.u != ScalarNormalizer.fit(subsample(te["u"], 2))


def test_torch_dataset_shapes_and_encoding():
    norms = DarcyNormalizers.from_json(require(NORMS).read_text())
    path = require(PROCESSED / "darcy_beta1.0_train.npz")
    ds64 = DarcyDataset(path, norms, stride=2, n=100)
    ds128 = DarcyDataset(path, norms, stride=1, n=10)
    x, y = ds64[0]
    assert len(ds64) == 100 and x.shape == y.shape == (1, 64, 64) and x.dtype.is_floating_point
    assert ds128[0][0].shape == (1, 128, 128)
    np.testing.assert_allclose(ds64.x[:10, 0].numpy(), ds128.x[:10, 0, ::2, ::2].numpy())
    raw = load_split_arrays(path)["u"][0, ::2, ::2]
    np.testing.assert_allclose(norms.u.decode(y[0].numpy().astype(np.float64)), raw, rtol=1e-5, atol=1e-7)


def test_reference_is_separate_from_official_targets():
    """The CG reference must be stored apart from, and never overwrite, the official PDEBench targets."""
    ref = np.load(require(PROCESSED / "darcy_beta1.0_test_reference.npz"))
    test = load_split_arrays(require(PROCESSED / "darcy_beta1.0_test.npz"))
    assert ref["indices"].tolist() == test["indices"].tolist()
    assert "NOT the official" in str(ref["note"])
    _, u_raw = read_samples(RAW, test["indices"])
    np.testing.assert_array_equal(test["u"], u_raw)  # official targets untouched, all 1000
    assert not np.array_equal(ref["u_reference"], test["u"])


def test_reference_solves_the_discrete_pde():
    """Stored references = converged float64 CG solutions rounded to float32.

    The float64 solve reaches a relative residual <= 1e-10. float32 rounding (~6e-8 relative) is amplified by the
    operator (~1/dx^2), so the residual of the STORED arrays is ~1e-4 (measured; docs/M1_AUDIT_REPORT.md).
    We therefore check the property that matters: stored == float64 solution to float32 precision.
    """
    from pdeflow.data.physics import solve_steady_state

    ref = np.load(require(PROCESSED / "darcy_beta1.0_test_reference.npz"))["u_reference"]
    test = load_split_arrays(require(PROCESSED / "darcy_beta1.0_test.npz"))
    u64, info = solve_steady_state(test["a"][:3], 1.0, 1 / 128)
    assert info["converged"] and info["max_relative_residual"] <= 1e-10
    assert relative_residual(test["a"][:3], u64, 1.0, 1 / 128).max() < 1e-9
    np.testing.assert_allclose(ref[:3], u64, rtol=2e-7, atol=0)
    stored_res = relative_residual(test["a"], ref.astype(np.float64), 1.0, 1 / 128)
    official_res = test["pde_residual"].astype(np.float64)
    assert stored_res.max() < 1e-3  # float32 rounding floor
    assert np.median(stored_res) < 0.2 * np.median(official_res)


def test_all_processed_files_are_official_targets():
    for split in ("train", "val"):
        d = load_split_arrays(require(PROCESSED / f"darcy_beta1.0_{split}.npz"))
        _, u_raw = read_samples(RAW, d["indices"])
        np.testing.assert_array_equal(d["u"], u_raw)


def test_target_sources_are_labelled():
    for split in ("train", "val", "test"):
        d = load_split_arrays(require(PROCESSED / f"darcy_beta1.0_{split}.npz"))
        assert "official PDEBench" in str(d["target_source"])
    ref = np.load(require(PROCESSED / "darcy_beta1.0_test_reference.npz"))
    assert "independent steady-state reference" in str(ref["target_source"])
