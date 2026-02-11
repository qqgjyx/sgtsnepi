"""Smoke tests for example workflows."""

import numpy as np
import pytest
from sklearn.datasets import load_digits

from pysgtsnepi import SGtSNEpi


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
