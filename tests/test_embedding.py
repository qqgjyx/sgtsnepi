"""Tests for embedding optimization loop."""

import numpy as np
from scipy.sparse import csc_matrix

from pysgtsnepi.embedding import sgtsne_embedding


def _make_ring_graph(n, k=5):
    """Create a well-conditioned symmetric graph (k-nearest on ring).

    Each node connects to its k nearest neighbors on a ring,
    with Gaussian-decaying weights. This produces a more realistic
    probability matrix than a simple ring graph.
    """
    rows, cols, vals = [], [], []
    for i in range(n):
        for offset in range(1, k + 1):
            j = (i + offset) % n
            w = np.exp(-0.5 * offset**2)
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w, w])
    P = csc_matrix((vals, (rows, cols)), shape=(n, n))
    # Normalize total to 1
    P.data /= P.sum()
    return P


def test_embedding_shape():
    """Embedding output has correct shape."""
    n = 50
    P = _make_ring_graph(n)
    Y = sgtsne_embedding(P, d=2, max_iter=10, h=0.7, random_state=0)

    assert Y.shape == (n, 2)


def test_embedding_finite():
    """Embedding values should all be finite."""
    n = 50
    P = _make_ring_graph(n)
    Y = sgtsne_embedding(P, d=2, max_iter=10, h=0.7, random_state=0)

    assert np.all(np.isfinite(Y))


def test_embedding_zero_mean():
    """Embedding should be approximately zero-mean."""
    n = 50
    P = _make_ring_graph(n)
    Y = sgtsne_embedding(P, d=2, max_iter=5, h=0.7, random_state=1)

    assert np.allclose(Y.mean(axis=0), 0, atol=1e-10)


def test_embedding_with_initial():
    """Embedding accepts initial coordinates."""
    n = 50
    P = _make_ring_graph(n)
    Y0 = np.random.default_rng(42).standard_normal((n, 2)) * 0.01
    Y = sgtsne_embedding(P, d=2, max_iter=5, h=0.7, Y0=Y0)

    assert Y.shape == (n, 2)
    assert np.all(np.isfinite(Y))
