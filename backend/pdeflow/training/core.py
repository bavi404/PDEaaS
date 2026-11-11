"""Loss, reproducibility helpers, checkpoints and the training loop."""

from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.normalization import DarcyNormalizers
from ..models.fno import build_model, count_parameters


# ---------------------------------------------------------------- reproducibility / environment
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- loss
def relative_l2_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of per-sample ||pred - target||_2 / ||target||_2 (physical units)."""
    b = pred_phys.shape[0]
    diff = (pred_phys - target_phys).reshape(b, -1).norm(dim=1)
    return (diff / target_phys.reshape(b, -1).norm(dim=1)).mean()


def decode(t: torch.Tensor, norms: DarcyNormalizers) -> torch.Tensor:
    return t * (norms.u.std + norms.u.eps) + norms.u.mean


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: dict, norms: DarcyNormalizers, extra: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "pdeflow-fno-v1", "model_state": model.state_dict(), "config": cfg,
                "normalizers": json.loads(norms.to_json()), **extra}, path)
    return sha256_file(path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, DarcyNormalizers, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") != "pdeflow-fno-v1":
        raise ValueError(f"unsupported checkpoint format: {ck.get('format')}")
    model = build_model(ck["config"]["model"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    norms = DarcyNormalizers.from_json(json.dumps(ck["normalizers"]))
    return model, norms, ck


# ---------------------------------------------------------------- evaluation helper
@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, norms: DarcyNormalizers, batch_size: int = 64,
            device: torch.device | str = "cpu") -> torch.Tensor:
    """x: encoded inputs (N,1,H,W). Returns DECODED predictions on CPU."""
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch_size):
        out.append(decode(model(x[i:i + batch_size].to(device)), norms).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- training loop
def train(cfg: dict, train_ds, val_ds, norms: DarcyNormalizers, run_dir: Path,
          log: Callable[[str, dict], None]) -> dict:
    tc = cfg["train"]
    seed_everything(cfg["seed"])
    torch.set_num_threads(int(tc.get("num_threads", 4)))
    device = resolve_device(tc.get("device", "auto"))
    use_amp = tc.get("amp", "auto") == "auto" and device.type == "cuda"
    model = build_model(cfg["model"]).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"]) if tc.get("scheduler") == "cosine" else None
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, generator=gen, drop_last=False)
    y_val = decode(val_ds.y, norms)
    log("train_start", {"device": str(device), "amp": use_amp, "n_params": n_params, "n_train": len(train_ds),
                        "n_val": len(val_ds), "resolution": train_ds.resolution, "torch": torch.__version__,
                        "threads": torch.get_num_threads()})

    best = {"val": math.inf, "epoch": -1}
    history = []
    t_start = time.monotonic()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0 = time.monotonic()
        tot, n, gmax, nonfinite = 0.0, 0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with ctx:
                pred = model(x)
            loss = relative_l2_loss(decode(pred.float(), norms), decode(y, norms))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"])
            if not torch.isfinite(gnorm) or not torch.isfinite(loss):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                continue  # skip a non-finite step (counted and logged; a run with any is flagged)
            opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
            gmax = max(gmax, float(gnorm))
        if sched:
            sched.step()
        train_loss = tot / max(n, 1)
        pv = predict(model, val_ds.x, norms, device=device)
        val = float(relative_l2_loss(pv, y_val))
        rec = {"epoch": epoch, "train_rel_l2": train_loss, "val_rel_l2_official_64": val, "lr": opt.param_groups[0]["lr"],
               "max_grad_norm": gmax, "nonfinite_steps": nonfinite, "epoch_s": round(time.monotonic() - t0, 2),
               "peak_rss_mb": round(peak_rss_mb(), 1)}
        history.append(rec)
        if val < best["val"]:
            best = {"val": val, "epoch": epoch}
            sha = save_checkpoint(run_dir / "best.pt", model, cfg, norms,
                                  {"epoch": epoch, "val_rel_l2_official_64": val, "n_params": n_params})
            rec["checkpoint_saved"] = sha[:16]
        log("epoch", rec)
    total = time.monotonic() - t_start
    save_checkpoint(run_dir / "last.pt", model, cfg, norms, {"epoch": tc["epochs"], "n_params": n_params})
    summary = {"best_epoch": best["epoch"], "best_val_rel_l2_official_64": best["val"], "train_time_s": round(total, 1),
               "mean_epoch_s": round(float(np.mean([h["epoch_s"] for h in history])), 2), "peak_rss_mb": round(peak_rss_mb(), 1),
               "n_params": n_params, "device": str(device), "amp": use_amp,
               "total_nonfinite_steps": int(sum(h["nonfinite_steps"] for h in history)),
               "best_checkpoint_sha256": sha256_file(run_dir / "best.pt")}
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    log("train_end", summary)
    return summary
