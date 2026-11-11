import numpy as np
import pytest

from pdeflow.data.normalization import DarcyNormalizers, ScalarNormalizer
from pdeflow.data.splits import SplitManifest, make_splits


def test_partition_and_sizes():
    m = make_splits(1000, seed=1)
    assert (len(m.train), len(m.val), len(m.test)) == (800, 100, 100)
    assert sorted(m.train + m.val + m.test) == list(range(1000))


def test_deterministic_and_seed_dependent():
    a, b, c = make_splits(500, 7), make_splits(500, 7), make_splits(500, 8)
    assert a.train == b.train and a.test == b.test
    assert a.test != c.test


def test_groups_stay_together():
    groups = list(range(100))
    for i in (3, 17, 42, 77, 91):
        groups[i] = "dup"
    for seed in range(20):
        m = make_splits(100, seed, groups=groups)
        homes = [n for n in ("train", "val", "test") if {3, 17, 42, 77, 91} & set(getattr(m, n))]
        assert len(homes) == 1
        assert m.duplicate_groups == [[3, 17, 42, 77, 91]]


def test_prefix_subset_is_stable():
    m = make_splits(1000, 3)
    assert m.subset("train", 50).tolist() == m.train[:50]
    with pytest.raises(ValueError):
        m.subset("val", 101)


def test_save_load_roundtrip(tmp_path):
    m = make_splits(200, 5, groups=[i // 2 for i in range(200)])
    m.save(tmp_path / "s.json")
    assert SplitManifest.load(tmp_path / "s.json") == m


def test_validate_detects_overlap():
    m = make_splits(100, 0)
    m.test = m.test[:-1] + [m.train[0]]
    with pytest.raises(ValueError):
        m.validate()


@pytest.mark.parametrize("fr", [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)])
def test_bad_fractions(fr):
    with pytest.raises(ValueError):
        make_splits(10, 0, fractions=fr)


def test_normalizer_roundtrip_and_stats():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(50, 8, 8))
    n = ScalarNormalizer.fit(x)
    assert n.mean == pytest.approx(x.mean()) and n.std == pytest.approx(x.std())
    np.testing.assert_allclose(n.decode(n.encode(x)), x, rtol=1e-12)
    z = n.encode(x)
    assert abs(z.mean()) < 1e-9 and z.std() == pytest.approx(1.0, rel=1e-6)


def test_normalizer_is_resolution_independent():
    n = ScalarNormalizer(mean=1.0, std=2.0)
    for r in (16, 64, 128):
        assert n.encode(np.ones((r, r))).shape == (r, r)


def test_normalizer_rejects_bad_input():
    with pytest.raises(ValueError):
        ScalarNormalizer.fit(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        ScalarNormalizer.fit(np.ones(10))


def test_normalizers_json_roundtrip():
    d = DarcyNormalizers(ScalarNormalizer(0.5, 0.4), ScalarNormalizer(0.16, 0.14), "unit test")
    assert DarcyNormalizers.from_json(d.to_json()) == d
