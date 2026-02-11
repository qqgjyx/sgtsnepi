"""Attractive forces for SG-t-SNE-Pi (sparse PQ component)."""

from __future__ import annotations

import numba
import numpy as np
from scipy.sparse import csc_matrix


@numba.njit(cache=True)
def _attractive_forces_kernel(
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    Y: np.ndarray,
    n: int,
    d: int,
) -> np.ndarray:
    """Compute attractive forces from sparse CSC matrix P and embedding Y.

    Translated from ref/sgtsnepi/src/pq.cpp.

    Parameters
    ----------
    indices : ndarray
        CSC row indices.
    indptr : ndarray
        CSC column pointers.
    data : ndarray
        CSC values (P_ij entries).
    Y : ndarray of shape (n, d)
        Current embedding coordinates.
    n : int
        Number of points.
    d : int
        Embedding dimensions.

    Returns
    -------
    ndarray of shape (n, d)
        Attractive force on each point.
    """
    Fattr = np.zeros((n, d), dtype=np.float64)

    for j in range(n):
        Yi = Y[j]
        for idx in range(indptr[j], indptr[j + 1]):
            i = indices[idx]
            # Squared distance in embedding space
            dist_sq = 0.0
            for dd in range(d):
                dist_sq += (Y[i, dd] - Yi[dd]) ** 2
            # P_ij * Q_ij where Q_ij = 1/(1 + ||y_i - y_j||^2)
            pq = data[idx] / (1.0 + dist_sq)
            # Accumulate force on point i
            for dd in range(d):
                Fattr[i, dd] += pq * (Y[i, dd] - Yi[dd])

    return Fattr


def attractive_forces(P: csc_matrix, Y: np.ndarray) -> np.ndarray:
    """Compute attractive forces from sparse probability matrix P.

    Parameters
    ----------
    P : csc_matrix of shape (n, n)
        Sparse probability matrix.
    Y : ndarray of shape (n, d)
        Current embedding coordinates.

    Returns
    -------
    ndarray of shape (n, d)
        Attractive force on each point.
    """
    n, d = Y.shape
    return _attractive_forces_kernel(
        P.indices, P.indptr, P.data, np.ascontiguousarray(Y), n, d
    )
