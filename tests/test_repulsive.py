"""Tests for repulsive forces module."""

import numpy as np

from pysgtsnepi.repulsive import (
    _best_grid_size,
    compute_repulsive_forces,
)


def test_best_grid_size():
    """Grid size selection returns valid FFT-friendly sizes."""
    assert _best_grid_size(10) >= 10
    assert _best_grid_size(14) >= 14
    # Result + 2 should be in the GRID_SIZES list
    from pysgtsnepi.repulsive import _GRID_SIZES

    result = _best_grid_size(20)
    assert (result + 2) in _GRID_SIZES


def test_repulsive_2d_shape():
    """2D repulsive forces have correct shape."""
    n, d = 50, 2
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((n, d))

    Frep, zeta = compute_repulsive_forces(Y, h=0.7)

    assert Frep.shape == (n, d)
    assert np.isfinite(zeta)
    assert zeta > 0


def test_repulsive_1d_shape():
    """1D repulsive forces have correct shape."""
    n, d = 30, 1
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((n, d))

    Frep, zeta = compute_repulsive_forces(Y, h=0.5)

    assert Frep.shape == (n, d)
    assert np.isfinite(zeta)


def test_repulsive_forces_finite():
    """All repulsive forces should be finite."""
    n, d = 40, 2
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((n, d)) * 5

    Frep, zeta = compute_repulsive_forces(Y, h=0.7)

    assert np.all(np.isfinite(Frep))
    assert np.isfinite(zeta)
