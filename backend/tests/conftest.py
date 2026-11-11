"""Shared fixtures.

Unit tests use a tiny *synthetic* HDF5 file that mimics the PDEBench Darcy layout. It exists only
to exercise code paths (ordering, validation errors); it is never used for training or reported
results. Tests marked ``data`` use the real downloaded PDEBench file and are skipped if it is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"
PROCESSED = REPO / "data/processed"
SPLITS = REPO / "data/splits/darcy_beta1.0_seed20261111.json"
NORMS = REPO / "data/splits/normalizers_beta1.0_train2000_stride2.json"


def write_fake_darcy(path: Path, n: int = 6, r: int = 8, beta: float = 1.0, **overrides) -> Path:
    rng = np.random.default_rng(0)
    nu = np.where(rng.random((n, r, r)) > 0.5, 1.0, 0.1).astype(np.float32)
    # encode the sample index in u so ordering can be checked
    u = (np.arange(n, dtype=np.float32)[:, None, None, None] + np.zeros((n, 1, r, r), np.float32))
    x = ((np.arange(r) + 0.5) / r).astype(np.float32)
    data = {"nu": nu, "tensor": u, "x-coordinate": x, "y-coordinate": x}
    data.update(overrides)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if v is not None:
                f.create_dataset(k, data=v)
        f.attrs["beta"] = beta
    return path


@pytest.fixture
def fake_h5(tmp_path: Path) -> Path:
    return write_fake_darcy(tmp_path / "fake.hdf5")


def require(path: Path) -> Path:
    """Skip if a real-data artefact is missing -- unless PDEFLOW_REQUIRE_DATA=1, where missing data FAILS.

    The final validation run uses PDEFLOW_REQUIRE_DATA=1 so that no real-data test can be silently skipped.
    """
    if not path.exists():
        msg = f"{path.relative_to(REPO)} not present (run scripts/download_pdebench.py and scripts/preprocess.py)"
        if os.environ.get("PDEFLOW_REQUIRE_DATA") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    return path
