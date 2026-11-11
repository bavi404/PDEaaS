"""Reader for PDEBench 2D Darcy Flow HDF5 files.

Verified layout of ``2D_DarcyFlow_beta1.0_Train.hdf5`` (see docs/DATA.md):

    nu            (N, 128, 128)    float32   permeability / diffusion coefficient a(x)
    tensor        (N, 1, 128, 128) float32   solution u(x) (single, final time slice)
    x-coordinate  (128,)           float32   cell centres (i + 0.5) / 128
    y-coordinate  (128,)           float32   cell centres (j + 0.5) / 128
    attrs: beta = 1.0                         constant forcing

The reader validates this layout on open and fails loudly on any deviation, so a different
or corrupted file can never be silently consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

INPUT_KEY = "nu"
TARGET_KEY = "tensor"
X_KEY = "x-coordinate"
Y_KEY = "y-coordinate"


class DatasetFormatError(ValueError):
    """Raised when an HDF5 file does not match the expected PDEBench Darcy layout."""


@dataclass(frozen=True)
class DarcyFileInfo:
    path: Path
    n_samples: int
    resolution: int
    beta: float
    x: np.ndarray
    y: np.ndarray


def inspect_file(path: str | Path) -> DarcyFileInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        missing = {INPUT_KEY, TARGET_KEY, X_KEY, Y_KEY} - set(f.keys())
        if missing:
            raise DatasetFormatError(f"missing datasets {sorted(missing)} in {path.name}")
        nu, u = f[INPUT_KEY], f[TARGET_KEY]
        if nu.ndim != 3 or nu.shape[1] != nu.shape[2]:
            raise DatasetFormatError(f"'{INPUT_KEY}' must be (N, R, R), got {nu.shape}")
        n, r = nu.shape[0], nu.shape[1]
        if u.shape != (n, 1, r, r):
            raise DatasetFormatError(f"'{TARGET_KEY}' must be ({n}, 1, {r}, {r}), got {u.shape}")
        if nu.dtype != np.float32 or u.dtype != np.float32:
            raise DatasetFormatError(f"expected float32, got {nu.dtype} / {u.dtype}")
        x, y = f[X_KEY][:], f[Y_KEY][:]
        if x.shape != (r,) or y.shape != (r,):
            raise DatasetFormatError("coordinate arrays do not match spatial resolution")
        if "beta" not in f.attrs:
            raise DatasetFormatError("root attribute 'beta' missing")
        beta = float(f.attrs["beta"])
    return DarcyFileInfo(path=path, n_samples=n, resolution=r, beta=beta, x=x, y=y)


def read_samples(path: str | Path, indices: np.ndarray | list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, u)`` with shapes ``(k, R, R)`` in float32, in the order of ``indices``.

    h5py fancy indexing requires increasing indices, so we read sorted and restore the order.
    """
    info = inspect_file(path)
    with h5py.File(info.path, "r") as f:
        if indices is None:
            return f[INPUT_KEY][:], f[TARGET_KEY][:, 0]
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("indices must be a non-empty 1-D sequence")
        if idx.min() < 0 or idx.max() >= info.n_samples:
            raise IndexError(f"indices out of range [0, {info.n_samples})")
        if np.unique(idx).size != idx.size:
            raise ValueError("indices contain duplicates")
        order = np.argsort(idx)
        sorted_idx = idx[order]
        a = f[INPUT_KEY][sorted_idx]
        u = f[TARGET_KEY][sorted_idx, 0]
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return a[inv], u[inv]
