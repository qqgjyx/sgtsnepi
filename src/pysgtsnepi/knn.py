"""kNN graph construction via PyNNDescent."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from pynndescent import NNDescent
from scipy.sparse import csc_matrix

logger = logging.getLogger(__name__)


def _perplexity_equalize(
    D_sq: csc_matrix,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> csc_matrix:
    """Convert squared distances to probabilities via perplexity equalization.

    Direct translation of Julia ``util.jl:perplexity_equalization``.

    For each column *j*, binary-search for ``sigma`` such that the
    Shannon entropy of the resulting distribution equals ``log(perplexity)``.

    Parameters
    ----------
    D_sq : csc_matrix
        Sparse matrix of squared distances (non-negative).
    perplexity : float
        Target perplexity.
    tol : float
        Convergence tolerance for bisection.
    max_iter : int
        Maximum bisection iterations per column.

    Returns
    -------
    csc_matrix
        Column-stochastic probability matrix.
    """
    P = csc_matrix(D_sq, dtype=np.float64, copy=True)
    n = P.shape[0]
    tiny = np.finfo(np.float64).tiny

    H = np.log(perplexity)  # target entropy
    sigma = np.ones(n)
    i_diff = np.zeros(n)
    i_count = np.zeros(n)

    for j in range(n):
        start, end = P.indptr[j], P.indptr[j + 1]
        if start == end:
            continue

        vals = D_sq.data[start:end]  # squared distances (read-only)

        # Compute initial entropy at sigma=1
        fval = _col_entropy(vals, 1.0, tiny) - H

        lb = float("-inf")
        ub = float("inf")
        it = 0

        # Bisection search (Julia lines 265-281)
        while abs(fval) > tol and it < max_iter:
            it += 1

            if fval > 0:
                lb = sigma[j]
                if np.isinf(ub):
                    sigma[j] = 2 * lb
                else:
                    sigma[j] = 0.5 * (lb + ub)
            else:
                ub = sigma[j]
                if np.isinf(lb):
                    sigma[j] = 0.5 * ub
                else:
                    sigma[j] = 0.5 * (lb + ub)

            fval = _col_entropy(vals, sigma[j], tiny) - H

        i_diff[j] = fval
        i_count[j] = it

        # Update column: normalized exp(-D² * sigma)  (Julia lines 214-218)
        exp_vals = np.exp(-vals * sigma[j])
        col_sum = exp_vals.sum()
        if col_sum > 0:
            exp_vals /= col_sum
        P.data[start:end] = exp_vals

    # Diagnostics
    avg_iter = int(np.ceil(np.sum(i_count) / max(n, 1)))
    nc_idx = int(np.sum(np.abs(i_diff) > tol))

    if nc_idx == 0:
        logger.info(
            "All %d elements converged numerically, avg(#iter) = %d", n, avg_iter
        )
    else:
        warnings.warn(
            f"There are {nc_idx} non-convergent elements out of {n}", stacklevel=2
        )

    n_neg = int(np.sum(sigma < 0))
    if n_neg > 0:
        warnings.warn(
            f"There are {n_neg} nodes with negative sigma;"
            " consider decreasing perplexity",
            stacklevel=2,
        )

    return P


def _col_entropy(vals: np.ndarray, sigma: float, tiny: float) -> float:
    """Compute Shannon entropy for a column.

    Matches Julia ``colentropy`` (util.jl lines 176-201):
      h = log(Z) + sigma * sum(D² * P)
    where Z = sum(exp(-D² * sigma)), P = exp(-D² * sigma) / Z.
    """
    exp_v = np.exp(-vals * sigma)
    sum_j = max(exp_v.sum(), tiny)
    h = np.log(sum_j) + sigma * np.sum(vals * exp_v) / sum_j
    return h


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int | None = None,
    perplexity: float | None = None,
) -> csc_matrix:
    """Build sparse kNN graph from a point cloud.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_neighbors : int
        Number of nearest neighbors (excluding self).
    metric : str
        Distance metric for PyNNDescent.
    random_state : int or None
        Random seed for reproducibility.
    perplexity : float or None
        If set, apply perplexity equalization (Gaussian kernel with
        adaptive bandwidth) and return a column-stochastic probability
        matrix.  If ``None``, return raw squared distances.

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse (n_samples, n_samples) matrix.  Column-stochastic
        probabilities if *perplexity* is set, raw squared distances
        otherwise.  Self-loops removed.
    """
    n = X.shape[0]
    k = n_neighbors

    # PyNNDescent finds k+1 neighbors (includes self at index 0)
    index = NNDescent(
        X,
        n_neighbors=k + 1,
        metric=metric,
        random_state=random_state,
    )
    indices, distances = index.neighbor_graph

    # Remove self-neighbor (first column is always self with distance ~0)
    knn_indices = indices[:, 1:]  # (n, k)
    knn_distances = distances[:, 1:]  # (n, k)

    # Square distances (algorithm expects D²)
    knn_distances = knn_distances**2

    # Build CSC sparse matrix: column j has k entries at rows knn_indices[j,:]
    row_ind = knn_indices.ravel()
    col_ind = np.repeat(np.arange(n), k)
    data = knn_distances.ravel()

    graph = csc_matrix((data, (row_ind, col_ind)), shape=(n, n))

    # Remove any remaining self-loops and explicit zeros
    graph.setdiag(0)
    graph.eliminate_zeros()

    # Apply perplexity equalization if requested
    if perplexity is not None:
        graph = _perplexity_equalize(graph, perplexity)

    return graph
