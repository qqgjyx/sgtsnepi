"""SG-t-SNE-Pi gradient descent optimization loop.

Translated from ref/sgtsnepi/src/gradient_descend.cpp.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from pysgtsnepi.attractive import attractive_forces
from pysgtsnepi.repulsive import compute_repulsive_forces


def sgtsne_embedding(
    P: csc_matrix,
    d: int = 2,
    max_iter: int = 1000,
    early_exag: int = 250,
    alpha: float = 12.0,
    eta: float = 200.0,
    h: float = 1.0,
    Y0: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Run SG-t-SNE-Pi gradient descent.

    Parameters
    ----------
    P : csc_matrix
        Symmetrized, normalized probability matrix (n, n).
    d : int
        Embedding dimensions (1, 2, or 3).
    max_iter : int
        Total iterations.
    early_exag : int
        Number of early exaggeration iterations.
    alpha : float
        Exaggeration multiplier (drops to 1 after early_exag).
    eta : float
        Learning rate.
    h : float
        Grid side length for FFT.
    Y0 : ndarray or None
        Initial embedding of shape (n, d). If None, random init
        with scale 0.01 (matching Julia tutorial convention).
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    Y : ndarray of shape (n, d)
        Final embedding coordinates.
    """
    n = P.shape[0]
    rng = np.random.default_rng(random_state)

    # Initialize embedding
    if Y0 is not None:
        Y = Y0.copy().astype(np.float64)
    else:
        Y = rng.standard_normal((n, d)) * 0.01

    # Momentum parameters (same as C++ reference)
    momentum = 0.5
    final_momentum = 0.8
    mom_switch_iter = 250

    # Allocate state
    uY = np.zeros((n, d), dtype=np.float64)
    gains = np.ones((n, d), dtype=np.float64)

    current_alpha = alpha

    for it in range(max_iter):
        # Compute attractive forces
        Fattr = attractive_forces(P, Y)

        # Compute repulsive forces
        Frep, _zeta = compute_repulsive_forces(Y, h)

        # Gradient: dY = alpha * Fattr - Frep
        dY = current_alpha * Fattr - Frep

        # Gain adaptation
        gains = np.where(np.sign(dY) != np.sign(uY), gains + 0.2, gains * 0.8)
        np.clip(gains, 0.01, None, out=gains)

        # Velocity update
        uY = momentum * uY - eta * gains * dY

        # Position update
        Y += uY

        # Zero-mean center
        Y -= Y.mean(axis=0)

        # Stop early exaggeration
        if it == early_exag:
            current_alpha = 1.0

        # Switch momentum
        if it == mom_switch_iter:
            momentum = final_momentum

    return Y
