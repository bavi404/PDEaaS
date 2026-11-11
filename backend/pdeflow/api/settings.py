"""API configuration from environment variables (no secrets are needed or read here).

| Variable                              | Default                                         |
|---------------------------------------|-------------------------------------------------|
| PDEFLOW_CHECKPOINT                    | backend/models/fno_darcy_beta1_baseline.pt      |
| PDEFLOW_EXPECTED_CHECKPOINT_SHA256    | the evaluated checkpoint (pinned below)         |
| PDEFLOW_SAMPLES_FILE                  | backend/assets/demo_samples_v1.npz              |
| PDEFLOW_EVAL_REPORT                   | reports/m2_eval_fno_baseline.json               |
| PDEFLOW_NUM_THREADS                   | 4                                               |
| PDEFLOW_CORS_ORIGINS                  | http://localhost:3000 (comma-separated; "" = none) |
| PDEFLOW_MAX_BODY_BYTES                | 1000000                                         |
| PDEFLOW_LOG_FILE                      | unset (stderr only)                             |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# SHA-256 of runs/fno_baseline/best.pt, the checkpoint evaluated in reports/m2_eval_fno_baseline.json.
EXPECTED_CHECKPOINT_SHA256 = "d7133c7c681cf3e9632579833fe357636d84ff7a5678d94b2f24c38955124df7"


def _path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


@dataclass(frozen=True)
class Settings:
    checkpoint: Path = field(default_factory=lambda: _path("PDEFLOW_CHECKPOINT",
                                                            REPO_ROOT / "backend/models/fno_darcy_beta1_baseline.pt"))
    expected_sha256: str = field(default_factory=lambda: os.environ.get("PDEFLOW_EXPECTED_CHECKPOINT_SHA256",
                                                                        EXPECTED_CHECKPOINT_SHA256))
    samples_file: Path = field(default_factory=lambda: _path("PDEFLOW_SAMPLES_FILE",
                                                              REPO_ROOT / "backend/assets/demo_samples_v1.npz"))
    eval_report: Path = field(default_factory=lambda: _path("PDEFLOW_EVAL_REPORT",
                                                             REPO_ROOT / "reports/m2_eval_fno_baseline.json"))
    num_threads: int = field(default_factory=lambda: int(os.environ.get("PDEFLOW_NUM_THREADS", "4")))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("PDEFLOW_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()))
    max_body_bytes: int = field(default_factory=lambda: int(os.environ.get("PDEFLOW_MAX_BODY_BYTES", "1000000")))
    log_file: str | None = field(default_factory=lambda: os.environ.get("PDEFLOW_LOG_FILE") or None)
