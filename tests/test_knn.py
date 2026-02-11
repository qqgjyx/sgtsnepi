"""Tests for knn module."""

import numpy as np
from scipy.sparse import issparse

from pysgtsnepi.knn import build_knn_graph


def test_basic_shape():
    """kNN graph has correct shape and sparsity."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    k = 10
    graph = build_knn_graph(X, n_neighbors=k, random_state=42)

    assert issparse(graph)
    assert graph.shape == (50, 50)
    # Each column should have at most k entries
    for j in range(50):
        col_nnz = graph.indptr[j + 1] - graph.indptr[j]
        assert col_nnz <= k


def test_no_self_loops():
    """Diagonal should be zero (no self-loops)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    graph = build_knn_graph(X, n_neighbors=5, random_state=0)

    diag = graph.diagonal()
    assert np.all(diag == 0)


def test_squared_distances():
    """All values should be non-negative (squared distances)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 4))
    graph = build_knn_graph(X, n_neighbors=8, random_state=1)

    assert np.all(graph.data >= 0)


def test_csc_format():
    """Output is CSC format."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 3))
    graph = build_knn_graph(X, n_neighbors=5, random_state=2)

    assert graph.format == "csc"
