"""Tests for attractive forces module."""

import numpy as np
from scipy.sparse import csc_matrix

from pysgtsnepi.attractive import attractive_forces


def test_output_shape():
    """Attractive forces have shape (n, d)."""
    n, d = 20, 2
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((n, d))
    # Simple sparse P
    row = np.array([1, 2, 3, 0, 2, 0])
    col = np.array([0, 0, 1, 1, 3, 3])
    data = np.array([0.5, 0.5, 0.3, 0.7, 0.5, 0.5])
    P = csc_matrix((data, (row, col)), shape=(n, n))

    Fattr = attractive_forces(P, Y)
    assert Fattr.shape == (n, d)


def test_zero_for_identity_positions():
    """If all points are at same position, forces should be zero."""
    n, d = 10, 2
    Y = np.zeros((n, d))
    row = [1, 2]
    col = [0, 0]
    data = [0.5, 0.5]
    P = csc_matrix((data, (row, col)), shape=(n, n))

    Fattr = attractive_forces(P, Y)
    assert np.allclose(Fattr, 0)


def test_forces_finite():
    """All forces should be finite."""
    n, d = 30, 3
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((n, d))
    # Ring graph
    row = [*range(1, n), 0]
    col = list(range(n))
    data = [1.0 / n] * n
    P = csc_matrix((data, (row, col)), shape=(n, n))

    Fattr = attractive_forces(P, Y)
    assert np.all(np.isfinite(Fattr))
