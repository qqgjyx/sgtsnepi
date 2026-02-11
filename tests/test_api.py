"""Tests for functional API."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from pysgtsnepi.api import sgtsnepi
from pysgtsnepi.knn import build_knn_graph


def test_api_shape():
    """API returns correct shape with kNN graph input."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    A = build_knn_graph(X, n_neighbors=10, random_state=0)

    Y = sgtsnepi(A, d=2, max_iter=10, random_state=42)

    assert Y.shape == (50, 2)
    assert np.all(np.isfinite(Y))


def test_api_lambda_1():
    """Lambda=1 skips rescaling."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    A = build_knn_graph(X, n_neighbors=10, random_state=0)

    Y = sgtsnepi(A, d=2, lambda_=1.0, max_iter=10, random_state=0)

    assert Y.shape == (30, 2)
    assert np.all(np.isfinite(Y))


def test_api_rejects_dense():
    """API rejects dense input."""
    with pytest.raises(TypeError):
        sgtsnepi(np.eye(10))


def test_api_rejects_nonsquare():
    """API rejects non-square matrix."""
    A = csc_matrix(np.ones((3, 5)))
    with pytest.raises(ValueError):
        sgtsnepi(A)
