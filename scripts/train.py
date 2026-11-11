#!/usr/bin/env python3
"""Train the FNO on the validated M1 pipeline.

    .venv/bin/python scripts/train.py --config backend/configs/fno_baseline.yaml
    .venv/bin/python scripts/train.py --config backend/configs/fno_baseline.yaml --smoke   # 64 train / 64 val, 3 epochs

Writes runs/<run_name>/{best.pt,last.pt,history.json,config.resolved.json,summary.json,train.jsonl}.
Training and model selection use OFFICIAL PDEBench targets only (train / val splits). The independent
steady-state reference is never loaded here.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import torch  # noqa: E402

from pdeflow.data.dataset import DarcyDataset  # noqa: E402
from pdeflow.data.normalization import DarcyNormalizers  # noqa: E402
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402
from pdeflow.training.core import sha256_file, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the loop: 64 train, 64 val, 3 epochs")
    ap.add_argument("--epochs", type=int, help="override train.epochs")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg = copy.deepcopy(cfg)
    if args.smoke:
        cfg["run_name"] += "_smoke"
        cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["train"]["epochs"] = 64, 64, 3
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"pdeflow.train.{cfg['run_name']}", run_dir / "train.jsonl")

    def log(event: str, fields: dict) -> None:
        log_event(logger, event, **fields)

    d = cfg["data"]
    norms = DarcyNormalizers.from_json(Path(d["normalizer_path"]).read_text())
    train_ds = DarcyDataset(d["train_file"], norms, stride=d["stride"], n=d["n_train"])
    val_ds = DarcyDataset(d["val_file"], norms, stride=d["stride"], n=d["n_val"])
    cfg["provenance"] = {
        "config_file": args.config, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_file_sha256": sha256_file(d["train_file"]), "val_file_sha256": sha256_file(d["val_file"]),
        "splits_sha256": sha256_file(d["splits_path"]), "normalizer_sha256": sha256_file(d["normalizer_path"]),
        "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(), "cuda_available": torch.cuda.is_available(),
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2))
    log("config", cfg)
    summary = train(cfg, train_ds, val_ds, norms, run_dir, log)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if summary["total_nonfinite_steps"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
