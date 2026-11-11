#!/usr/bin/env python3
"""Create splits, processed split files and train-only normalisation statistics.

Steps (each logged to logs/m1_preprocess.jsonl):
  1. verify the raw file's MD5 against its provenance record (written by download_pdebench.py)
  2. group byte-identical inputs and build the seeded, group-aware 80/10/10 split manifest
  3. write processed .npz files (native 128x128 float32 + per-sample PDE residual) for the
     configured train prefix and the full val/test splits
  4. fit scalar normalisers on the training prefix at the training stride ONLY
  5. write reports/preprocessing_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from pdeflow.data.normalization import DarcyNormalizers, ScalarNormalizer  # noqa: E402
from pdeflow.data.pdebench import inspect_file, read_samples  # noqa: E402
from pdeflow.data.physics import relative_residual  # noqa: E402
from pdeflow.data.resample import subsample  # noqa: E402
from pdeflow.data.splits import make_splits  # noqa: E402
from pdeflow.logging_utils import get_logger, log_event  # noqa: E402


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="backend/configs/data.yaml")
    ap.add_argument("--log", default="logs/m1_preprocess.jsonl")
    ap.add_argument("--summary", default="reports/preprocessing_summary.json")
    args = ap.parse_args()
    log = get_logger("pdeflow.preprocess", args.log)
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_event(log, "config_loaded", **cfg)

    src = Path(cfg["source_file"])
    prov = json.loads(Path(cfg["provenance_file"]).read_text())
    md5 = file_md5(src)
    if md5 != prov["md5"]:
        log_event(log, "md5_mismatch", 40, md5=md5, expected=prov["md5"])
        return 1
    log_event(log, "md5_verified", md5=md5, darus_file_id=prov["darus_file_id"])

    info = inspect_file(src)
    a_all, u_all = read_samples(src)
    keys = [hashlib.md5(x.tobytes()).hexdigest() for x in a_all] if cfg["group_identical_inputs"] else None
    m = make_splits(info.n_samples, cfg["seed"], tuple(cfg["fractions"]),
                    source_file=src.name, source_md5=md5, groups=keys)
    m.save(cfg["splits_path"])
    dup_home = {}
    for g in m.duplicate_groups:
        dup_home[str(len(g))] = next(n for n in ("train", "val", "test") if set(g) & set(getattr(m, n)))
    log_event(log, "splits_created", path=cfg["splits_path"], seed=m.seed, sizes={"train": len(m.train), "val": len(m.val), "test": len(m.test)},
              group_policy=m.group_policy, duplicate_group_split=dup_home)

    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dx = 1.0 / info.resolution
    summary: dict = {"source_md5": md5, "seed": m.seed, "split_sizes_full": {"train": len(m.train), "val": len(m.val), "test": len(m.test)},
                     "duplicate_group_split": dup_home, "files": {}}
    for split, n in (("train", cfg["n_train"]), ("val", cfg["n_val"]), ("test", cfg["n_test"])):
        idx = m.subset(split, n)
        a, u = a_all[idx], u_all[idx]
        res = relative_residual(a, u, info.beta, dx).astype(np.float32)
        path = out_dir / f"darcy_beta{info.beta}_{split}.npz"
        np.savez(path, a=a, u=u, indices=idx, pde_residual=res, beta=np.float32(info.beta),
                 target_source=np.array("official PDEBench `tensor`, unmodified (training/benchmark target)"))
        n_dup = int(np.isin(idx, [i for g in m.duplicate_groups for i in g]).sum())
        summary["files"][split] = {"path": str(path), "n": int(idx.size), "residual_median": float(np.median(res)),
                                   "residual_frac_above_0.01": float((res > 0.01).mean()), "n_duplicate_group_members": n_dup,
                                   "u_mean": float(u.mean()), "u_std": float(u.std()), "a_frac_high": float((a == a.max()).mean())}
        log_event(log, "split_file_written", split=split, **summary["files"][split])

    # train-only normalisation at the training resolution
    tr = np.load(out_dir / f"darcy_beta{info.beta}_train.npz")
    s = cfg["train_stride"]
    a_tr, u_tr = subsample(tr["a"], s), subsample(tr["u"], s)
    norms = DarcyNormalizers(a=ScalarNormalizer.fit(a_tr), u=ScalarNormalizer.fit(u_tr),
                             fitted_on=f"train[:{cfg['n_train']}] stride {s} ({a_tr.shape[-1]}x{a_tr.shape[-1]}), seed {m.seed}")
    Path(cfg["normalizer_path"]).write_text(norms.to_json())
    summary["normalizers"] = json.loads(norms.to_json())
    log_event(log, "normalizers_fitted", **summary["normalizers"])

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    log_event(log, "summary_written", path=args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
