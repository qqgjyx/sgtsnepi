"""Local edge weighting for unweighted graphs.

Translates ``local_weights.jl`` from the Julia SGtSNEpi.jl reference.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix


def local_weights(A: csc_matrix) -> csc_matrix:
    """Compute Jaccard-index edge weights for an unweighted graph.

    For each edge (i, j), the weight is set to:

    .. math::

        w_{ij} = \\frac{|N(i) \\cap N(j)|}{d_i + d_j - |N(i) \\cap N(j)|} + \\varepsilon

    where *N(v)* is the neighbor set, *d(v)* the degree, and
    :math:`\\varepsilon` is machine epsilon.

    Parameters
    ----------
    A : csc_matrix
        Binary (unweighted) adjacency matrix.

    Returns
    -------
    C : csc_matrix
        Weighted adjacency matrix with Jaccard-index weights.
    """
    A_bin = (A != 0).astype(np.float64)
    degrees = np.asarray(A_bin.sum(axis=0)).ravel()

    # A_bin^T @ A_bin gives intersection counts for all connected pairs
    intersection = A_bin.T @ A_bin  # sparse (n, n)

    # Build output — iterate over nonzeros of A
    C = A.copy().astype(np.float64)
    rows, cols = C.nonzero()

    # Vectorized: extract intersection values for all edges at once
    cap = np.asarray(intersection[rows, cols]).ravel()
    denom = degrees[rows] + degrees[cols] - cap
    C_data = cap / denom + np.finfo(np.float64).eps

    # Write back into sparse matrix
    # Build new csc from COO to ensure correct ordering
    C = csc_matrix((C_data, (rows, cols)), shape=A.shape)

    return C
