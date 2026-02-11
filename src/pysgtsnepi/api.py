"""Functional API for SG-t-SNE-Pi embedding."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, diags, issparse

from pysgtsnepi.embedding import sgtsne_embedding
from pysgtsnepi.utils.sgtsne_lambda_equalization import sgtsne_lambda_equalization


def sgtsnepi(
    A,
    d: int = 2,
    lambda_: float = 10.0,
    max_iter: int = 1000,
    early_exag: int = 250,
    alpha: float = 12.0,
    eta: float = 200.0,
    h: float = 0.0,
    Y0: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Embed sparse stochastic graph via SG-t-SNE-Pi.

    Parameters
    ----------
    A : sparse matrix
        Adjacency or stochastic matrix (n, n). Must be square.
    d : int
        Embedding dimensions (1, 2, or 3).
    lambda_ : float
        Rescaling parameter (default 10).
    max_iter : int
        Maximum iterations.
    early_exag : int
        Early exaggeration iterations.
    alpha : float
        Exaggeration multiplier.
    eta : float
        Learning rate.
    h : float
        Grid side length for FFT. If <= 0, auto-selected.
    Y0 : ndarray or None
        Initial embedding of shape (n, d).
    random_state : int or None
        Random seed.

    Returns
    -------
    Y : ndarray of shape (n, d)
        Embedding coordinates.
    """
    if d not in {1, 2, 3}:
        raise ValueError(f"d must be 1, 2, or 3, got {d}")

    if not issparse(A):
        raise TypeError("Input A must be a sparse matrix")

    P = csc_matrix(A, dtype=np.float64)
    n = P.shape[0]
    if P.shape[0] != P.shape[1]:
        raise ValueError("Input matrix must be square")

    # Remove self-loops
    P.setdiag(0)
    P.eliminate_zeros()

    # Validate
    if P.data.min() < 0:
        raise ValueError("Input matrix must have non-negative weights")

    # Track isolated nodes (zero-column)
    col_sums = np.asarray(P.sum(axis=0)).ravel()
    active = col_sums > 0
    n_active = active.sum()

    if n_active < 2:
        raise ValueError("Need at least 2 connected nodes")

    # Remove isolated nodes
    if n_active < n:
        active_idx = np.where(active)[0]
        P = P[np.ix_(active_idx, active_idx)]
        if Y0 is not None:
            Y0 = Y0[active_idx]
        col_sums = np.asarray(P.sum(axis=0)).ravel()

    # Make column-stochastic
    col_sums_safe = np.where(col_sums > 0, col_sums, 1.0)
    D_inv = 1.0 / col_sums_safe
    P = P @ diags(D_inv)

    # Lambda rescaling
    if lambda_ != 1.0:
        # Convert to distances: D = -log(P)
        D = P.copy()
        D.data = -np.log(np.maximum(D.data, np.finfo(float).tiny))
        D = sgtsne_lambda_equalization(D, lambda_)
        # Re-normalize columns to sum to 1
        cs = np.asarray(D.sum(axis=0)).ravel()
        cs_safe = np.where(cs > 0, cs, 1.0)
        P = D @ diags(1.0 / cs_safe)

    # Symmetrize: P = P + P.T (matches C++ sparsematrix.cpp)
    P = P + P.T

    # Normalize total: P /= P.sum()
    total = P.sum()
    if total > 0:
        P.data /= total

    P = csc_matrix(P)

    # Auto-select h
    if h <= 0:
        h = {1: 0.5, 2: 0.7, 3: 1.2}[d]

    # Run embedding
    Y = sgtsne_embedding(
        P,
        d=d,
        max_iter=max_iter,
        early_exag=early_exag,
        alpha=alpha,
        eta=eta,
        h=h,
        Y0=Y0,
        random_state=random_state,
    )

    # Place isolated nodes
    if n_active < n:
        Y_full = np.zeros((n, d), dtype=np.float64)
        Y_full[np.where(active)[0]] = Y
        # Place isolated nodes far from embedding
        if Y.size > 0:
            corner = Y.max(axis=0) + 1.0
            Y_full[~active] = corner
        Y = Y_full

    return Y
