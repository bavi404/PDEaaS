"""Model factory. One entry point so the API, training and evaluation build identical networks."""

from __future__ import annotations

import torch
from neuralop.models import FNO


def build_model(cfg: dict) -> torch.nn.Module:
    """Build neuraloperator's FNO from the ``model`` section of a config (library defaults otherwise)."""
    return FNO(
        n_modes=tuple(cfg["n_modes"]),
        hidden_channels=int(cfg["hidden_channels"]),
        n_layers=int(cfg["n_layers"]),
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 1)),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
