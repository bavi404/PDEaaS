"""Resolution changes between the native 128x128 PDEBench grid and coarser training grids.

Primary method: point subsampling with stride s (``x[..., ::s, ::s]``), identical to PDEBench's
own ``reduced_resolution`` option. It keeps the binary permeability values exact and introduces
no interpolation error in the targets. See docs/PREPROCESSING.md for the comparison against 2x2
average pooling and the cross-resolution validation.
"""

from __future__ import annotations

import numpy as np


def subsample(x: np.ndarray, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if x.shape[-1] % stride or x.shape[-2] % stride:
        raise ValueError(f"spatial shape {x.shape[-2:]} not divisible by stride {stride}")
    return np.ascontiguousarray(x[..., ::stride, ::stride])


def avg_pool(x: np.ndarray, factor: int) -> np.ndarray:
    """Block average (the exact cell-average restriction for finite-volume data)."""
    *lead, h, w = x.shape
    if h % factor or w % factor:
        raise ValueError("shape not divisible by factor")
    return x.reshape(*lead, h // factor, factor, w // factor, factor).mean(axis=(-3, -1))


def cell_centres(n_fine: int, stride: int = 1) -> np.ndarray:
    """Physical coordinates of the points kept by ``subsample`` on a cell-centred [0,1] grid."""
    return ((np.arange(n_fine) + 0.5) / n_fine)[::stride]


def _interp_axis(x: np.ndarray, src: np.ndarray, dst: np.ndarray, axis: int) -> np.ndarray:
    """Piecewise-linear interpolation along ``axis`` from coordinates ``src`` to ``dst`` (constant extrapolation)."""
    dst_c = np.clip(dst, src[0], src[-1])
    hi = np.clip(np.searchsorted(src, dst_c, side="right"), 1, len(src) - 1)
    lo = hi - 1
    w = (dst_c - src[lo]) / (src[hi] - src[lo])
    shape = [1] * x.ndim
    shape[axis] = -1
    w = w.reshape(shape)
    return np.take(x, lo, axis=axis) * (1 - w) + np.take(x, hi, axis=axis) * w


def linear_upsample(x: np.ndarray, src_coords: np.ndarray, dst_coords: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of (..., n, n) data sampled at ``src_coords`` onto ``dst_coords`` (both axes)."""
    y = _interp_axis(x, src_coords, dst_coords, axis=x.ndim - 2)
    return _interp_axis(y, src_coords, dst_coords, axis=x.ndim - 1)


def high_frequency_energy_fraction(x: np.ndarray, cutoff: int) -> np.ndarray:
    """Per-sample fraction of spectral energy with max(|kx|,|ky|) > cutoff (integer wavenumbers)."""
    X = np.fft.fft2(x, axes=(-2, -1))
    n = x.shape[-1]
    k = np.abs(np.fft.fftfreq(n, d=1.0 / n))
    mask = np.maximum(k[:, None], k[None, :]) > cutoff
    e = np.abs(X) ** 2
    return e[..., mask].sum(axis=-1) / e.sum(axis=(-2, -1))
