"""Model factory, loss, training loop and checkpoint round-trip on tiny generated tensors (no real data)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from pdeflow.data.normalization import DarcyNormalizers, ScalarNormalizer
from pdeflow.evaluation.metrics import relative_l2
from pdeflow.models.fno import build_model, count_parameters
from pdeflow.training.core import load_checkpoint, predict, relative_l2_loss, save_checkpoint, train

CFG = Path(__file__).resolve().parents[1] / "configs/fno_baseline.yaml"
TINY = {"n_modes": [4, 4], "hidden_channels": 8, "n_layers": 2, "in_channels": 1, "out_channels": 1}
NORMS = DarcyNormalizers(ScalarNormalizer(0.5, 0.45), ScalarNormalizer(0.16, 0.14), "unit test")


def test_baseline_config_builds_expected_model():
    cfg = yaml.safe_load(CFG.read_text())
    m = build_model(cfg["model"])
    assert count_parameters(m) == 357_217  # matches docs/BLOCKERS.md benchmark config A
    with torch.no_grad():
        for r in (64, 128):
            assert m(torch.randn(2, 1, r, r)).shape == (2, 1, r, r)


def test_loss_matches_numpy_metric():
    rng = np.random.default_rng(0)
    p, t = rng.random((4, 1, 8, 8)) + 1, rng.random((4, 1, 8, 8)) + 1
    torch_loss = relative_l2_loss(torch.tensor(p), torch.tensor(t)).item()
    assert torch_loss == pytest.approx(relative_l2(p[:, 0], t[:, 0]).mean(), rel=1e-12)


def _tiny_data(n=16, r=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, 1, r, r, generator=g) > 0.5).float() * 0.9 + 0.1
    # smooth, input-dependent positive target (a learnable map; not physics)
    u = torch.nn.functional.avg_pool2d(a, 5, 1, 2) * 0.3 + 0.05
    ds = SimpleNamespace(x=NORMS.a.encode(a), y=NORMS.u.encode(u), resolution=r)
    ds.__len__ = lambda: n
    return torch.utils.data.TensorDataset(ds.x, ds.y), ds


def _cfg(epochs, tmp):
    return {"seed": 0, "model": TINY, "train": {"epochs": epochs, "batch_size": 4, "lr": 3e-3, "weight_decay": 0.0,
            "scheduler": "cosine", "grad_clip_norm": 10.0, "amp": "auto", "device": "cpu", "num_threads": 1}}


class _DS(torch.utils.data.TensorDataset):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.x, self.y, self.resolution = x, y, r


def _run(tmp_path, epochs=25):
    _, d = _tiny_data()
    tr = _DS(d.x[:12], d.y[:12], 16)
    va = _DS(d.x[12:], d.y[12:], 16)
    events = []
    summary = train(_cfg(epochs, tmp_path), tr, va, NORMS, tmp_path, lambda e, f: events.append((e, f)))
    hist = json.loads((tmp_path / "history.json").read_text())
    return summary, hist, va


def test_training_decreases_loss_with_finite_gradients(tmp_path):
    summary, hist, _ = _run(tmp_path, epochs=60)  # 12-sample toy problem plateaus ~epoch 20, reaches ~0.41x by 60
    assert summary["total_nonfinite_steps"] == 0
    assert all(np.isfinite(h["max_grad_norm"]) and h["max_grad_norm"] > 0 for h in hist)
    assert hist[-1]["train_rel_l2"] < 0.5 * hist[0]["train_rel_l2"]
    assert summary["best_val_rel_l2_official_64"] <= min(h["val_rel_l2_official_64"] for h in hist) + 1e-12
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()


def test_training_is_deterministic_on_cpu(tmp_path):
    _, h1, _ = _run(tmp_path / "a", epochs=3)
    _, h2, _ = _run(tmp_path / "b", epochs=3)
    assert [h["train_rel_l2"] for h in h1] == [h["train_rel_l2"] for h in h2]


def test_best_checkpoint_roundtrip_reproduces_selection_metric(tmp_path):
    summary, _, va = _run(tmp_path, epochs=5)
    model, norms, ck = load_checkpoint(tmp_path / "best.pt")
    assert norms == NORMS and ck["epoch"] == summary["best_epoch"]
    pv = predict(model, va.x, norms)
    val = relative_l2_loss(pv, va.y * (norms.u.std + norms.u.eps) + norms.u.mean).item()
    assert val == pytest.approx(summary["best_val_rel_l2_official_64"], rel=1e-6)


def test_checkpoint_rejects_unknown_format(tmp_path):
    torch.save({"format": "other"}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "x.pt")


def test_save_load_identical_outputs(tmp_path):
    torch.manual_seed(0)
    m = build_model(TINY).eval()
    save_checkpoint(tmp_path / "c.pt", m, {"model": TINY}, NORMS, {"epoch": 0})
    m2, _, _ = load_checkpoint(tmp_path / "c.pt")
    x = torch.randn(2, 1, 16, 16)
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))
