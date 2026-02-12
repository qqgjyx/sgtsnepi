"""Functional API for SG-t-SNE-Pi embedding."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, diags, issparse

from pysgtsnepi.embedding import sgtsne_embedding
from pysgtsnepi.utils.graph_rescaling import lambda_rescaling
from pysgtsnepi.utils.graph_weights import local_weights


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
    unweighted_to_weighted: bool = True,
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
        Grid side length for FFT. If <= 0, defaults to 1.0
        (matching Julia wrapper).
    Y0 : ndarray or None
        Initial embedding of shape (n, d).
    random_state : int or None
        Random seed.
    unweighted_to_weighted : bool
        If True (default) and all edge weights are 1.0, compute
        Jaccard-index local weights before normalization. Matches
        Julia ``flag_unweighted_to_weighted``.

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

    # Convert unweighted to weighted (Jaccard index), matching Julia default
    if unweighted_to_weighted and np.all(P.data == 1.0):
        P = local_weights(P)

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

    # Lambda rescaling (C++ lambdaRescaling: -log, bisection, exp, col-normalize)
    if lambda_ != 1.0:
        P = lambda_rescaling(P, lambda_)

    # Symmetrize: P = P + P.T (matches C++ sparsematrix.cpp)
    P = P + P.T

    # Normalize total: P /= P.sum()
    total = P.sum()
    if total > 0:
        P.data /= total

    P = csc_matrix(P)

    # Auto-select h
    if h <= 0:
        h = 1.0

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
