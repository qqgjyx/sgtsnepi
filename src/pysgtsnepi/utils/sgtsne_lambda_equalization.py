import warnings
from typing import Literal

import numpy as np
from scipy.optimize import root_scalar
from scipy.sparse import csc_matrix


def _colsum(D, j, sigma=1.0):
    """Compute column sum of exp(-vals * sigma)."""
    D_min = np.finfo(float).tiny
    vals = D.data[D.indptr[j] : D.indptr[j + 1]]
    sum_j = np.sum(np.exp(-vals * sigma))
    return max(sum_j, D_min)


def _colupdate(D, j, sigma):
    """Update column values in-place with exp(-vals * sigma)."""
    start, end = D.indptr[j], D.indptr[j + 1]
    D.data[start:end] = np.exp(-D.data[start:end] * sigma)


def _make_objective(D, j, lambda_):
    """Create objective function for column j."""

    def objective(x):
        return _colsum(D, j, x) - lambda_

    return objective


def sgtsne_lambda_equalization(
    D: csc_matrix,
    lambda_: float,
    max_iter: int = 50,
    tol_binary: float = 1e-5,
    algorithm: Literal[
        "custom_bisection",
        "bisection",
        "brentq",
        "brenth",
        "bisect",
        "ridder",
        "newton",
        "secant",
        "halley",
    ] = "custom_bisection",
) -> csc_matrix:
    """Binary search for the scales of column-wise conditional probabilities.

    Binary search for the scales of column-wise conditional probabilities
    from exp(-D) to exp(-D/σ²)/z equalized by λ.

    Parameters
    ----------
    D : scipy.sparse.csc_matrix
        N x N sparse matrix of "distance square"
        (column-wise conditional, local distances)
    lambda_ : float
        The equalization parameter
    max_iter : int, optional
        Maximum number of iterations for binary search, by default 50
    tol_binary : float, optional
        Tolerance for binary search convergence, by default 1e-5
    algorithm : str, optional
        The root finding algorithm to use, by default "custom_bisection"

    Returns
    -------
    scipy.sparse.csc_matrix
        The column-wise conditional probability matrix

    Notes
    -----
    .. versionadded:: 0.1.0

    Author
    ------
    Xiaobai Sun (MATLAB prototype on May 12, 2019)
    Dimitris Floros (translation to Julia)
    Juntang Wang (translation to Python on Nov 16, 2024)
    """

    n = D.shape[0]
    cond_P = D.copy()

    i_diff = np.zeros(n)
    i_count = np.zeros(n)
    i_tval = np.zeros(n)
    sigma_sq = np.ones(n)

    # Pre-calculate initial column sums
    for j in range(n):
        sum_j = _colsum(D, j)
        i_tval[j] = sum_j - lambda_

    if algorithm == "custom_bisection":
        for j in range(n):
            fval = i_tval[j]
            lb, ub = -1000.0, 1000.0

            iter_count = 0

            while abs(fval) > tol_binary and iter_count < max_iter:
                iter_count += 1

                if fval > 0:
                    lb = sigma_sq[j]
                    if ub >= 1000.0:
                        sigma_sq[j] = 2 * lb
                    else:
                        sigma_sq[j] = 0.5 * (lb + ub)
                else:
                    ub = sigma_sq[j]
                    if lb <= -1000.0:
                        sigma_sq[j] = 0.5 * ub
                    else:
                        sigma_sq[j] = 0.5 * (lb + ub)

                sum_j = _colsum(D, j, sigma_sq[j])
                fval = sum_j - lambda_

            i_diff[j] = fval
            i_count[j] = iter_count
            _colupdate(cond_P, j, sigma_sq[j])

    else:
        for j in range(n):
            objective = _make_objective(D, j, lambda_)

            try:
                if algorithm in ["brentq", "brenth", "bisect", "ridder"]:
                    result = root_scalar(
                        objective,
                        method=algorithm,
                        bracket=[-1000.0, 1000.0],
                        xtol=tol_binary,
                        maxiter=max_iter,
                        full_output=True,
                    )
                elif algorithm in ["newton", "secant", "halley"]:
                    result = root_scalar(
                        objective,
                        method=algorithm,
                        x0=1.0,
                        xtol=tol_binary,
                        maxiter=max_iter,
                        full_output=True,
                    )
                else:
                    raise ValueError(f"Unsupported root finding method: {algorithm}")

                sigma_sq[j] = result.root
                i_diff[j] = objective(sigma_sq[j])
                i_count[j] = result.iterations
                _colupdate(cond_P, j, sigma_sq[j])

            except (ValueError, RuntimeError) as e:
                msg = f"Failed for column {j} with {algorithm} method: {e!s}"
                warnings.warn(msg, stacklevel=2)
                sigma_sq[j] = 1.0
                i_diff[j] = objective(sigma_sq[j])
                i_count[j] = 0
                _colupdate(cond_P, j, sigma_sq[j])

    avg_iter = np.ceil(np.sum(i_count) / n)
    nc_idx = np.sum(np.abs(i_diff) > tol_binary)

    if nc_idx == 0:
        print(f"All {n} elements converged numerically, avg(#iter) = {avg_iter}")
    else:
        warnings.warn(
            f"There are {nc_idx} non-convergent elements out of {n}", stacklevel=2
        )

    n_neg = np.sum(sigma_sq < 0)
    if n_neg > 0:
        warnings.warn(
            f"There are {n_neg} nodes with negative gamma_i;"
            " consider decreasing lambda",
            stacklevel=2,
        )

    return cond_P
