"""Smoke tests for example workflows."""

import numpy as np
import pytest
from sklearn.datasets import load_digits

from pysgtsnepi import SGtSNEpi
from pysgtsnepi.api import sgtsnepi


@pytest.fixture()
def digits_subset():
    """Load a small subset of digits for fast testing."""
    digits = load_digits()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(digits.data), size=200, replace=False)
    return digits.data[idx], digits.target[idx]


def test_digits_quickstart_smoke(digits_subset):
    """Smoke test: digits workflow runs and produces valid output."""
    X, _y = digits_subset
    model = SGtSNEpi(d=2, lambda_=10, n_neighbors=15, max_iter=10, random_state=0)
    Y = model.fit_transform(X)

    assert Y.shape == (200, 2)
    assert np.all(np.isfinite(Y))


def test_graph_embedding_smoke():
    """Smoke test: graph-input workflow runs and produces valid output."""
    from scipy.sparse import random as sp_random

    # Random sparse symmetric graph (no kNN needed)
    A = sp_random(50, 50, density=0.2, random_state=42)
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()

    Y = sgtsnepi(A, d=2, max_iter=10, random_state=0)
    assert Y.shape == (50, 2)
    assert np.all(np.isfinite(Y))
