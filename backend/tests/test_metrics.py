import numpy as np
import pytest

from pdeflow.evaluation.metrics import PRIMARY, SECONDARY, evaluate, relative_l2


def test_relative_l2_values():
    t = np.ones((2, 4, 4))
    p = t.copy()
    p[1] *= 1.1
    np.testing.assert_allclose(relative_l2(p, t), [0.0, 0.1])


def test_relative_l2_errors():
    with pytest.raises(ValueError):
        relative_l2(np.ones((1, 4, 4)), np.ones((1, 4, 5)))
    with pytest.raises(ValueError):
        relative_l2(np.ones((1, 4, 4)), np.zeros((1, 4, 4)))


def test_primary_and_secondary_are_separate():
    rng = np.random.default_rng(0)
    official = rng.random((5, 8, 8)) + 1
    reference = official * 1.05
    pred = official.copy()
    out = evaluate(pred, official, reference)
    assert out[PRIMARY]["relative_l2"]["max"] == 0.0
    assert out[SECONDARY]["relative_l2"]["mean"] == pytest.approx(1 / 1.05 * 0.05, rel=1e-9)
    assert set(out) == {PRIMARY, SECONDARY}


def test_secondary_absent_without_reference():
    official = np.ones((3, 4, 4))
    assert set(evaluate(official, official)) == {PRIMARY}


def test_strata():
    official = np.ones((4, 4, 4))
    pred = official.copy()
    pred[2:] *= 1.2
    out = evaluate(pred, official, strata={"converged": np.array([1, 1, 0, 0], bool), "empty": np.zeros(4, bool)})
    s = out[PRIMARY]["strata"]
    assert s["converged"]["max"] == 0.0 and "empty" not in s
