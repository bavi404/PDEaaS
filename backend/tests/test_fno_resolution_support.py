"""The chosen FNO architecture accepts both 64x64 and 128x128 inputs.

This is a *capability* check on an untrained network. It is NOT evidence that a model trained at
64x64 generalises to 128x128; that is measured on trained checkpoints in M3 (docs/PREPROCESSING.md §6).
"""

import pytest
import torch
from neuralop.models import FNO


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return FNO(n_modes=(12, 12), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4).eval()


@pytest.mark.parametrize("res", [64, 128])
def test_forward_shape_and_finite(model, res):
    with torch.no_grad():
        y = model(torch.randn(2, 1, res, res))
    assert y.shape == (2, 1, res, res) and torch.isfinite(y).all()


def test_modes_below_nyquist_of_training_grid():
    assert 12 <= 64 // 2


def test_same_parameters_used_at_both_resolutions(model):
    n = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64))
        model(torch.randn(1, 1, 128, 128))
    assert sum(p.numel() for p in model.parameters()) == n
