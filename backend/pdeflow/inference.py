"""Inference shared by the API and the offline verification (scripts/verify_api_inference.py).

This module performs EXACTLY the steps that produced reports/m2_eval_fno_baseline.json (scripts/evaluate.py):
    a (float32, physical) -> train-only scalar normaliser encode -> FNO -> decode (u normaliser)
and for 128x128 input either
    "fno128_native"      : the FNO applied directly at 128x128 (zero-shot; test mean rel-L2 0.0421), or
    "fno64_interpolated" : stride-2 subsample to the 64x64 training grid, FNO, then coordinate-aware bilinear
                           interpolation to the 128 cell centres (test mean rel-L2 0.0320; docs/RESULTS.md §2).
`scripts/verify_api_inference.py` re-runs the whole 1,000-sample test split through this class and checks that
the report's numbers are reproduced, so the service is verifiably the evaluated model.

The checkpoint is identified by SHA-256 and loading fails if it differs from the expected (pinned) hash.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pdeflow.data.resample import cell_centres, linear_upsample, subsample
from pdeflow.training.core import load_checkpoint, predict, sha256_file

TRAINING_RESOLUTION = 64
SUPPORTED_RESOLUTIONS = (64, 128)  # the two resolutions with measured test error (docs/RESULTS.md)
STRATEGIES = ("fno_native", "fno64_interpolated", "fno128_native")
# Measured on the test split (docs/RESULTS.md §2): at 128x128, interpolating the 64x64 prediction (0.0320) beats
# zero-shot 128x128 inference (0.0421), so it is the default.
DEFAULT_STRATEGY = {64: "fno_native", 128: "fno64_interpolated"}
# Permeability values seen in training (binary fields, PDEBench Darcy). Anything else is out of distribution.
TRAINING_VALUES = (0.1, 1.0)


class ChecksumMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    u: np.ndarray  # (N, n, n) float64, physical units
    strategy: str
    resolution: int
    model_resolution: int
    inference_ms: float


class Predictor:
    def __init__(self, checkpoint: str | Path, expected_sha256: str | None, num_threads: int = 4):
        self.checkpoint = Path(checkpoint)
        self.sha256 = sha256_file(self.checkpoint)
        if expected_sha256 is not None and self.sha256 != expected_sha256.lower():
            raise ChecksumMismatch(f"{self.checkpoint}: sha256 {self.sha256} != expected {expected_sha256}")
        torch.set_num_threads(num_threads)
        self.num_threads = num_threads
        self.model, self.norms, self.ck = load_checkpoint(self.checkpoint)
        self._lock = threading.Lock()  # one forward at a time: avoids CPU thread oversubscription

    # ----------------------------------------------------------------- metadata
    def info(self) -> dict:
        ck = self.ck
        return {
            "format": ck["format"], "sha256": self.sha256, "file": self.checkpoint.name,
            "architecture": "neuraloperator FNO (2D)", "model_config": ck["config"]["model"],
            "n_params": ck["n_params"], "epoch": ck["epoch"],
            "selection_metric": {"name": "val_rel_l2_official_64", "value": ck["val_rel_l2_official_64"]},
            "training_resolution": TRAINING_RESOLUTION, "supported_resolutions": list(SUPPORTED_RESOLUTIONS),
            "default_strategy": {str(k): v for k, v in DEFAULT_STRATEGY.items()},
            "normalizers": {"a": self.norms.a.to_dict(), "u": self.norms.u.to_dict(),
                            "fitted_on": self.norms.fitted_on},
            "training_config": {k: ck["config"][k] for k in ("seed", "train", "data") if k in ck["config"]},
            "provenance": ck["config"].get("provenance", {}),
            "num_threads": self.num_threads,
        }

    # ----------------------------------------------------------------- inference
    def _forward(self, a: np.ndarray) -> np.ndarray:
        """Identical to scripts/evaluate.py `run_model`: float32 encode -> predict -> float64."""
        x = torch.from_numpy(self.norms.a.encode(a.astype(np.float32)).astype(np.float32)).unsqueeze(1)
        with self._lock:
            out = predict(self.model, x, self.norms)
        return out.numpy()[:, 0].astype(np.float64)

    def predict(self, a: np.ndarray, strategy: str | None = None) -> Prediction:
        """``a``: (n, n) or (N, n, n) permeability in physical units, n in SUPPORTED_RESOLUTIONS."""
        single = a.ndim == 2
        a = a[None] if single else a
        if a.ndim != 3 or a.shape[1] != a.shape[2]:
            raise ValueError(f"expected (N, n, n) or (n, n), got {a.shape}")
        n = a.shape[-1]
        if n not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"resolution {n} not supported; use one of {SUPPORTED_RESOLUTIONS}")
        strategy = strategy or DEFAULT_STRATEGY[n]
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy}")
        if (n == 64) != (strategy == "fno_native"):
            raise ValueError(f"strategy {strategy} is not valid for {n}x{n} input")
        t0 = time.perf_counter()
        if strategy == "fno64_interpolated":
            u64 = self._forward(subsample(a, 2))
            u = linear_upsample(u64, cell_centres(n, 2), cell_centres(n))
            model_res = TRAINING_RESOLUTION
        else:
            u = self._forward(a)
            model_res = n
        ms = (time.perf_counter() - t0) * 1000
        return Prediction(u[0] if single else u, strategy, n, model_res, ms)
