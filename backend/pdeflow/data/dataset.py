"""Torch dataset over a preprocessed split file (native 128x128 arrays, strided on access)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import DarcyNormalizers
from .resample import subsample


def load_split_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    for k in ("a", "u", "indices"):
        if k not in out:
            raise KeyError(f"{path}: missing array '{k}'")
    if out["a"].shape != out["u"].shape or out["a"].shape[0] != out["indices"].shape[0]:
        raise ValueError(f"{path}: inconsistent shapes {out['a'].shape} {out['u'].shape} {out['indices'].shape}")
    return out


class DarcyDataset(Dataset):
    """Yields ``(x, y)`` float32 tensors of shape (1, R/s, R/s), normalised with train-only statistics.

    ``n`` takes a prefix of the split (splits are pre-shuffled, so a prefix is a random subset).
    """

    def __init__(self, path: str | Path, normalizers: DarcyNormalizers, stride: int = 1, n: int | None = None):
        arrs = load_split_arrays(path)
        a, u, idx = arrs["a"], arrs["u"], arrs["indices"]
        if n is not None:
            if n > a.shape[0]:
                raise ValueError(f"requested {n} samples, split has {a.shape[0]}")
            a, u, idx = a[:n], u[:n], idx[:n]
        a, u = subsample(a, stride), subsample(u, stride)
        self.indices = idx
        self.stride = stride
        self.resolution = a.shape[-1]
        self.normalizers = normalizers
        self.x = torch.from_numpy(normalizers.a.encode(a).astype(np.float32)).unsqueeze(1)
        self.y = torch.from_numpy(normalizers.u.encode(u).astype(np.float32)).unsqueeze(1)
        n_all = arrs["indices"].shape[0]
        self.extra = {k: (v[: len(idx)] if v.ndim >= 1 and v.shape[0] == n_all else v)
                      for k, v in arrs.items() if k not in ("a", "u", "indices")}

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]
