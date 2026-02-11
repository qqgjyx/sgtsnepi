"""Tests for sklearn-compatible estimator."""

import numpy as np

from pysgtsnepi.estimator import SGtSNEpi


def test_fit_transform_dense():
    """Estimator works with dense input."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    model = SGtSNEpi(d=2, max_iter=10, n_neighbors=10, random_state=42)
    Y = model.fit_transform(X)

    assert Y.shape == (50, 2)
    assert np.all(np.isfinite(Y))
    assert hasattr(model, "embedding_")
    np.testing.assert_array_equal(Y, model.embedding_)


def test_fit_transform_sparse():
    """Estimator works with sparse adjacency input."""
    # Build kNN graph from data (realistic sparse matrix)
    from pysgtsnepi.knn import build_knn_graph

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    A = build_knn_graph(X, n_neighbors=10, random_state=0)

    model = SGtSNEpi(d=2, max_iter=10, random_state=0)
    Y = model.fit_transform(A)

    assert Y.shape == (50, 2)
    assert np.all(np.isfinite(Y))


def test_get_params():
    """Estimator get_params works (sklearn compatibility)."""
    model = SGtSNEpi(d=3, lambda_=5.0, n_neighbors=20)
    params = model.get_params()

    assert params["d"] == 3
    assert params["lambda_"] == 5.0
    assert params["n_neighbors"] == 20


def test_set_params():
    """Estimator set_params works (sklearn compatibility)."""
    model = SGtSNEpi()
    model.set_params(d=3, max_iter=500)

    assert model.d == 3
    assert model.max_iter == 500
