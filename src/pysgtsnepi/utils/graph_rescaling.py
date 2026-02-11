"""Lambda-based graph rescaling.

Direct translation of C++ ``graph_rescaling.cpp:lambdaRescaling``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


def lambda_rescaling(
    P: csc_matrix,
    lambda_: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> csc_matrix:
    """Rescale column-stochastic graph using lambda parameter.

    For each column *j* of the input matrix (which must already be
    column-stochastic):

    1. Convert probabilities to distances: ``D = -log(P)``
    2. Binary-search for ``sigma`` such that
       ``sum(exp(-D * sigma)) == lambda_``
    3. Update values: ``P_ij = exp(-D_ij * sigma)``
    4. Normalize column to sum to 1

    This matches C++ ``graph_rescaling.cpp:lambdaRescaling`` (lines 19-103).

    Parameters
    ----------
    P : csc_matrix
        Column-stochastic sparse matrix (values in (0, 1]).
    lambda_ : float
        Target column sum after rescaling (before normalization).
    tol : float
        Convergence tolerance for bisection.
    max_iter : int
        Maximum bisection iterations per column.

    Returns
    -------
    csc_matrix
        Rescaled column-stochastic matrix.
    """
    P = csc_matrix(P, dtype=np.float64, copy=True)
    n = P.shape[0]

    sigma = np.ones(n)
    i_diff = np.zeros(n)
    i_count = np.zeros(n)

    # Convert probabilities to distances in-place: D = -log(P)
    # Matches C++ line 40: P.val[j] = -log(P.val[j])
    tiny = np.finfo(np.float64).tiny
    P.data[:] = -np.log(np.maximum(P.data, tiny))

    for j in range(n):
        start, end = P.indptr[j], P.indptr[j + 1]
        if start == end:
            continue

        vals = P.data[start:end]

        # Initial residual: sum(exp(-D * 1.0)) - lambda
        lse = logsumexp(-vals)
        fval = (np.exp(lse) if lse < 700 else float("inf")) - lambda_

        lb = -1e3
        ub = float("inf")
        it = 0

        # Bisection search (C++ lines 54-84)
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
                sigma[j] = 0.5 * (lb + ub)

            lse = logsumexp(-vals * sigma[j])
            sum_j = np.exp(lse) if lse < 700 else float("inf")
            if sum_j == 0:
                sum_j = tiny
            fval = sum_j - lambda_

        i_diff[j] = fval
        i_count[j] = it

        # Update values: exp(-D * sigma)  (C++ lines 89-92)
        new_vals = np.exp(-vals * sigma[j])
        # Column-stochastic normalization (C++ lines 95-96)
        col_sum = new_vals.sum()
        if col_sum > 0:
            new_vals /= col_sum
        P.data[start:end] = new_vals

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
            f"There are {n_neg} nodes with negative sigma; consider decreasing lambda",
            stacklevel=2,
        )

    return P
