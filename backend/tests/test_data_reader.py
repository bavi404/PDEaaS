import numpy as np
import pytest

from pdeflow.data.pdebench import DatasetFormatError, inspect_file, read_samples
from tests.conftest import write_fake_darcy


def test_inspect_valid_file(fake_h5):
    info = inspect_file(fake_h5)
    assert (info.n_samples, info.resolution, info.beta) == (6, 8, 1.0)


def test_read_preserves_requested_order(fake_h5):
    a, u = read_samples(fake_h5, [4, 0, 2])
    assert a.shape == (3, 8, 8) and u.shape == (3, 8, 8)
    np.testing.assert_array_equal(u[:, 0, 0], [4, 0, 2])


def test_read_all(fake_h5):
    a, u = read_samples(fake_h5)
    assert a.shape == u.shape == (6, 8, 8)


@pytest.mark.parametrize("bad", [[6], [-1], [1, 1], []])
def test_read_rejects_bad_indices(fake_h5, bad):
    with pytest.raises((IndexError, ValueError)):
        read_samples(fake_h5, bad)


def test_missing_key(tmp_path):
    p = write_fake_darcy(tmp_path / "f.h5", tensor=None)
    with pytest.raises(DatasetFormatError, match="missing"):
        inspect_file(p)


def test_wrong_target_shape(tmp_path):
    p = write_fake_darcy(tmp_path / "f.h5", tensor=np.zeros((6, 8, 8), np.float32))
    with pytest.raises(DatasetFormatError, match="tensor"):
        inspect_file(p)


def test_wrong_dtype(tmp_path):
    p = write_fake_darcy(tmp_path / "f.h5", nu=np.ones((6, 8, 8), np.float64))
    with pytest.raises(DatasetFormatError, match="float32"):
        inspect_file(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_file(tmp_path / "nope.h5")
