"""Scalar Gaussian normalisation fitted on the training split only.

Scalar (global) mean/std rather than per-grid-point statistics: per-point statistics are tied
to one grid and cannot be applied at another resolution without interpolation, whereas a scalar
transform is resolution-independent, which is required for cross-resolution evaluation and for
an API that may accept other grid sizes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: float
    std: float
    eps: float = 1e-8

    @classmethod
    def fit(cls, x: np.ndarray) -> "ScalarNormalizer":
        x64 = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x64).all():
            raise ValueError("non-finite values in normalisation data")
        std = float(x64.std())
        if std == 0.0:
            raise ValueError("zero variance")
        return cls(mean=float(x64.mean()), std=std)

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScalarNormalizer":
        return cls(**d)


@dataclass(frozen=True)
class DarcyNormalizers:
    a: ScalarNormalizer
    u: ScalarNormalizer
    fitted_on: str  # description of the data used, e.g. "train[:2000] @ stride 2"

    def to_json(self) -> str:
        return json.dumps({"a": self.a.to_dict(), "u": self.u.to_dict(), "fitted_on": self.fitted_on}, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DarcyNormalizers":
        d = json.loads(s)
        return cls(a=ScalarNormalizer.from_dict(d["a"]), u=ScalarNormalizer.from_dict(d["u"]), fitted_on=d["fitted_on"])
