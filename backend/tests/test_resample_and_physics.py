import numpy as np
import pytest

from pdeflow.data.physics import apply_operator, fv_residual, solve_steady_state
from pdeflow.data.resample import avg_pool, cell_centres, high_frequency_energy_fraction, linear_upsample, subsample


def test_subsample_values_and_coords():
    x = np.arange(16 * 16, dtype=float).reshape(16, 16)
    s = subsample(x, 2)
    assert s.shape == (8, 8)
    np.testing.assert_array_equal(s, x[::2, ::2])
    np.testing.assert_allclose(cell_centres(16, 2), (4 * np.arange(8) + 1) / 32)
    with pytest.raises(ValueError):
        subsample(np.zeros((15, 15)), 2)


def test_embedding_offset_consistent_for_stride_only():
    """neuralop embeds point j of an n-grid at j/n; true coordinates differ by a constant only for stride."""
    off128 = cell_centres(128) - np.arange(128) / 128
    off64s = cell_centres(128, 2) - np.arange(64) / 64
    off64a = cell_centres(64) - np.arange(64) / 64
    assert np.allclose(off128, off128[0]) and np.allclose(off64s, off128[0])
    assert not np.allclose(off64a, off128[0])


def test_avg_pool():
    x = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(avg_pool(x, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_linear_upsample_exact_for_bilinear_functions():
    src, dst = cell_centres(32, 2), cell_centres(32)
    X, Y = np.meshgrid(src, src, indexing="ij")
    Xd, Yd = np.meshgrid(dst, dst, indexing="ij")
    f = lambda x, y: 2 * x - 3 * y + 0.5 * x * y + 1  # noqa: E731
    up = linear_upsample(f(X, Y)[None], src, dst)[0]
    inside = (dst >= src[0]) & (dst <= src[-1])
    np.testing.assert_allclose(up[np.ix_(inside, inside)], f(Xd, Yd)[np.ix_(inside, inside)], atol=1e-12)


def test_high_frequency_fraction():
    n = 64
    x = np.arange(n) / n
    lo = np.sin(2 * np.pi * 3 * x)[:, None] * np.ones(n)[None]
    hi = np.sin(2 * np.pi * 20 * x)[:, None] * np.ones(n)[None]
    assert high_frequency_energy_fraction(lo[None], 12)[0] < 1e-20
    assert high_frequency_energy_fraction(hi[None], 12)[0] == pytest.approx(1.0)


def _rand_a(n, r, seed=0):
    return np.where(np.random.default_rng(seed).random((n, r, r)) > 0.5, 1.0, 0.1)


def test_operator_symmetric_positive_definite():
    a = _rand_a(1, 16)[0]
    rng = np.random.default_rng(1)
    v, w = rng.normal(size=(2, 16, 16))
    dx = 1 / 16
    assert np.sum(v * apply_operator(a, w, dx)) == pytest.approx(np.sum(w * apply_operator(a, v, dx)), rel=1e-10)
    assert np.sum(v * apply_operator(a, v, dx)) > 0


def test_cg_solution_has_zero_residual_and_scales_with_beta():
    a = _rand_a(3, 24)
    dx = 1 / 24
    u1, info = solve_steady_state(a, 1.0, dx, tol=1e-12)
    assert info["converged"]
    assert np.abs(fv_residual(a, u1, 1.0, dx)).max() < 1e-8
    assert (u1 > 0).all()  # maximum principle: positive source, zero boundary
    u2, _ = solve_steady_state(a, 2.0, dx, tol=1e-12)
    np.testing.assert_allclose(u2, 2 * u1, rtol=1e-8)


def test_cg_constant_coefficient_matches_continuum_on_effective_domain():
    """a = 1: zero ghost values put u = 0 at the ghost-cell centres, i.e. on a square of side 1 + dx.

    The continuum centre value of -Lap u = 1 scales with side^2, so the discrete solution must match
    ref * (1 + dx)^2 to second order (this pins down the boundary interpretation in docs/DATA.md).
    """
    ref = 0.07367135  # series solution of -Lap u = 1, u = 0 on the unit square, at the centre
    errs = []
    for r in (16, 32, 64):
        u, _ = solve_steady_state(np.ones((r, r)), 1.0, 1 / r, tol=1e-12)
        centre = u[r // 2 - 1: r // 2 + 1, r // 2 - 1: r // 2 + 1].mean()  # 4 cells around the centre
        errs.append(abs(centre - ref * (1 + 1 / r) ** 2) / ref)
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3
    assert errs[1] / errs[2] > 3.0  # ~second-order convergence


def test_cg_rejects_nonpositive_permeability():
    with pytest.raises(ValueError):
        solve_steady_state(np.zeros((8, 8)), 1.0, 1 / 8)
