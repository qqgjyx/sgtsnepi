"""FFT-accelerated repulsive forces for SG-t-SNE-Pi.

Translated from ref/sgtsnepi/src/qq.cpp, gridding.cpp, nuconv.cpp,
non_periodic_conv.cpp.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from scipy.fft import fftn, ifftn

# FFT-friendly grid sizes (from qq.cpp)
_GRID_SIZES = [
    8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 26, 28, 32, 33, 35,
    36, 39, 40, 42, 44, 45, 48, 49, 50, 52, 54, 55, 56, 60, 63, 64,
    65, 66, 70, 72, 75, 77, 78, 80, 84, 88, 90, 91, 96, 98, 99, 100,
    104, 105, 108, 110, 112, 117, 120, 125, 126, 130, 132, 135, 140,
    144, 147, 150, 154, 156, 160, 165, 168, 175, 176, 180, 182, 189,
    192, 195, 196, 198, 200, 208, 210, 216, 220, 224, 225, 231, 234,
    240, 245, 250, 252, 260, 264, 270, 273, 275, 280, 288, 294, 297,
    300, 308, 312, 315, 320, 325, 330, 336, 343, 350, 351, 352, 360,
    364, 375, 378, 385, 390, 392, 396, 400, 416, 420, 432, 440, 441,
    448, 450, 455, 462, 468, 480, 490, 495, 500, 504, 512,
]


def _best_grid_size(n_grid: int) -> int:
    """Find smallest FFT-friendly size >= n_grid + 2, then subtract 2."""
    target = n_grid + 2
    for gs in _GRID_SIZES:
        if target <= gs:
            return gs - 2
    return _GRID_SIZES[-1] - 2


# ---- Lagrange interpolation kernels (degree 3) ----

@numba.njit(cache=True)
def _lagrange_inner(d):
    return 0.5 * d * d * d - d * d - 0.5 * d + 1.0


@numba.njit(cache=True)
def _lagrange_outer(d):
    cc = 1.0 / 6.0
    return -cc * d * d * d + d * d - 11.0 * cc * d + 1.0


# ---- Scatter to Grid (S2G) ----

@numba.njit(cache=True)
def _s2g_1d(V, y, q, ng, n, n_vec):
    """Scatter values from points to 1D grid."""
    for i in range(n):
        f1 = math.floor(y[i, 0])
        d = y[i, 0] - f1
        w1 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])
        for j in range(n_vec):
            qv = q[i, j]
            for idx1 in range(4):
                V[f1 + idx1, j] += qv * w1[idx1]


@numba.njit(cache=True)
def _s2g_2d(V, y, q, ng, n, n_vec):
    """Scatter values from points to 2D grid."""
    for i in range(n):
        f1 = math.floor(y[i, 0])
        d = y[i, 0] - f1
        w1 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        f2 = math.floor(y[i, 1])
        d = y[i, 1] - f2
        w2 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        for j in range(n_vec):
            qv = q[i, j]
            for idx2 in range(4):
                qv2 = qv * w2[idx2]
                for idx1 in range(4):
                    V[f1 + idx1, f2 + idx2, j] += qv2 * w1[idx1]


@numba.njit(cache=True)
def _s2g_3d(V, y, q, ng, n, n_vec):
    """Scatter values from points to 3D grid."""
    for i in range(n):
        f1 = math.floor(y[i, 0])
        d = y[i, 0] - f1
        w1 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        f2 = math.floor(y[i, 1])
        d = y[i, 1] - f2
        w2 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        f3 = math.floor(y[i, 2])
        d = y[i, 2] - f3
        w3 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        for j in range(n_vec):
            qv = q[i, j]
            for idx3 in range(4):
                for idx2 in range(4):
                    qv23 = qv * w2[idx2] * w3[idx3]
                    for idx1 in range(4):
                        V[f1 + idx1, f2 + idx2, f3 + idx3, j] += (
                            qv23 * w1[idx1]
                        )


# ---- Grid to Scatter (G2S) ----

@numba.njit(cache=True)
def _g2s_1d(Phi, V, y, ng, n, n_vec):
    """Gather values from 1D grid to scattered points."""
    for i in range(n):
        f1 = math.floor(y[i, 0])
        d = y[i, 0] - f1
        w1 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])
        for j in range(n_vec):
            accum = 0.0
            for idx1 in range(4):
                accum += V[f1 + idx1, j] * w1[idx1]
            Phi[i, j] = accum


@numba.njit(cache=True)
def _g2s_2d(Phi, V, y, ng, n, n_vec):
    """Gather values from 2D grid to scattered points."""
    for i in range(n):
        f1 = math.floor(y[i, 0])
        d = y[i, 0] - f1
        w1 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        f2 = math.floor(y[i, 1])
        d = y[i, 1] - f2
        w2 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        for j in range(n_vec):
            accum = 0.0
            for idx2 in range(4):
                for idx1 in range(4):
                    accum += V[f1 + idx1, f2 + idx2, j] * w2[idx2] * w1[idx1]
            Phi[i, j] = accum


@numba.njit(cache=True)
def _g2s_3d(Phi, V, y, ng, n, n_vec):
    """Gather values from 3D grid to scattered points."""
    for i in range(n):
        f1 = math.floor(y[i, 0])
        d = y[i, 0] - f1
        w1 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        f2 = math.floor(y[i, 1])
        d = y[i, 1] - f2
        w2 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        f3 = math.floor(y[i, 2])
        d = y[i, 2] - f3
        w3 = np.array([
            _lagrange_outer(1 + d), _lagrange_inner(d),
            _lagrange_inner(1 - d), _lagrange_outer(2 - d),
        ])

        for j in range(n_vec):
            accum = 0.0
            for idx3 in range(4):
                for idx2 in range(4):
                    w23 = w2[idx2] * w3[idx3]
                    for idx1 in range(4):
                        accum += (
                            V[f1 + idx1, f2 + idx2, f3 + idx3, j]
                            * w23
                            * w1[idx1]
                        )
            Phi[i, j] = accum


# ---- FFT convolution (even/odd decomposition) ----

def _conv1d_nopad(PhiGrid, VGrid, h, ng, n_vec):
    """Non-periodic 1D convolution via even/odd decomposition (2 passes)."""
    hsq = h * h
    n1 = ng

    # Twiddle factors: w[i] = exp(-2*pi*i*j / (2*n1))
    w = np.exp(-2j * np.pi * np.arange(n1) / (2 * n1))

    # Build kernel values K[i] = (1 + hsq * i^2)^{-2}
    idx = np.arange(n1)
    kvals = (1.0 + hsq * idx * idx) ** (-2)

    # ---- EVEN pass ----
    K = kvals.astype(complex).copy()
    K[n1 - 1:0:-1] += kvals[1:]

    X = VGrid.copy().astype(complex)  # shape (n1, n_vec)

    K_hat = fftn(K, axes=(0,))
    X_hat = fftn(X, axes=(0,))
    X_hat *= K_hat[:, np.newaxis]
    result_even = ifftn(X_hat, axes=(0,)).real

    # ---- ODD pass ----
    K = kvals.astype(complex).copy()
    K[n1 - 1:0:-1] -= kvals[1:]
    K *= w

    X = VGrid.astype(complex) * w[:, np.newaxis]

    K_hat = fftn(K, axes=(0,))
    X_hat = fftn(X, axes=(0,))
    X_hat *= K_hat[:, np.newaxis]
    X_inv = ifftn(X_hat, axes=(0,))
    X_inv *= np.conj(w)[:, np.newaxis]
    result_odd = X_inv.real

    # scipy ifftn already divides by N, so just apply 1/2^d factor
    PhiGrid[:] = (result_even + result_odd) * 0.5


def _conv2d_nopad(PhiGrid, VGrid, h, ng, n_vec):
    """Non-periodic 2D convolution via even/odd decomposition (4 passes)."""
    hsq = h * h
    n1 = n2 = ng

    # Twiddle factors (only along dim 0 in C++ — but used for both dims)
    w = np.exp(-2j * np.pi * np.arange(n1) / (2 * n1))

    # Kernel values
    ii = np.arange(n1)
    jj = np.arange(n2)
    gi, gj = np.meshgrid(ii, jj, indexing="ij")
    kvals = (1.0 + hsq * (gi * gi + gj * gj)) ** (-2)

    # Each pass: build kernel with symmetry, apply twiddle, FFT convolve
    # Symmetry signs for (dim1, dim2): e=+1, o=-1
    passes = [
        (1, 1, None, None),       # ee
        (-1, 1, w, None),         # oe: twiddle on dim 0
        (1, -1, None, w),         # eo: twiddle on dim 1
        (-1, -1, w, w),           # oo: twiddle on both
    ]

    accum = np.zeros((n1, n2, n_vec), dtype=np.float64)

    for s1, s2, tw1, tw2 in passes:
        K = kvals.astype(complex).copy()
        K[n1 - 1:0:-1, :] += s1 * kvals[1:, :]
        K[:, n2 - 1:0:-1] += s2 * kvals[:, 1:]
        K[n1 - 1:0:-1, n2 - 1:0:-1] += s1 * s2 * kvals[1:, 1:]

        # Apply twiddle to kernel
        if tw1 is not None:
            K *= tw1[:, np.newaxis]
        if tw2 is not None:
            K *= tw2[np.newaxis, :]

        # Setup RHS with twiddle
        X = VGrid.astype(complex)
        if tw1 is not None:
            X *= tw1[:, np.newaxis, np.newaxis]
        if tw2 is not None:
            X *= tw2[np.newaxis, :, np.newaxis]

        # FFT, Hadamard, IFFT
        K_hat = fftn(K, axes=(0, 1))
        X_hat = fftn(X, axes=(0, 1))
        X_hat *= K_hat[:, :, np.newaxis]
        X_inv = ifftn(X_hat, axes=(0, 1))

        # Un-twiddle
        if tw1 is not None:
            X_inv *= np.conj(tw1)[:, np.newaxis, np.newaxis]
        if tw2 is not None:
            X_inv *= np.conj(tw2)[np.newaxis, :, np.newaxis]

        accum += X_inv.real

    # scipy ifftn already divides by N, so just apply 1/2^d factor
    PhiGrid[:] = accum * 0.25


def _conv3d_nopad(PhiGrid, VGrid, h, ng, n_vec):
    """Non-periodic 3D convolution via even/odd decomposition (8 passes)."""
    hsq = h * h
    n1 = n2 = n3 = ng

    w = np.exp(-2j * np.pi * np.arange(n1) / (2 * n1))

    ii = np.arange(n1)
    gi, gj, gk = np.meshgrid(ii, ii, ii, indexing="ij")
    kvals = (1.0 + hsq * (gi * gi + gj * gj + gk * gk)) ** (-2)

    accum = np.zeros((n1, n2, n3, n_vec), dtype=np.float64)

    # 8 passes: all combinations of even/odd for 3 dims
    for s1 in (1, -1):
        for s2 in (1, -1):
            for s3 in (1, -1):
                K = kvals.astype(complex).copy()
                K[n1 - 1:0:-1, :, :] += s1 * kvals[1:, :, :]
                K[:, n2 - 1:0:-1, :] += s2 * kvals[:, 1:, :]
                K[n1 - 1:0:-1, n2 - 1:0:-1, :] += s1 * s2 * kvals[1:, 1:, :]
                K[:, :, n3 - 1:0:-1] += s3 * kvals[:, :, 1:]
                K[n1 - 1:0:-1, :, n3 - 1:0:-1] += s1 * s3 * kvals[1:, :, 1:]
                K[:, n2 - 1:0:-1, n3 - 1:0:-1] += s2 * s3 * kvals[:, 1:, 1:]
                K[n1 - 1:0:-1, n2 - 1:0:-1, n3 - 1:0:-1] += (
                    s1 * s2 * s3 * kvals[1:, 1:, 1:]
                )

                # Twiddle: odd dims get twiddle factor
                tw1 = w if s1 == -1 else None
                tw2 = w if s2 == -1 else None
                tw3 = w if s3 == -1 else None

                if tw1 is not None:
                    K *= tw1[:, np.newaxis, np.newaxis]
                if tw2 is not None:
                    K *= tw2[np.newaxis, :, np.newaxis]
                if tw3 is not None:
                    K *= tw3[np.newaxis, np.newaxis, :]

                X = VGrid.astype(complex)
                if tw1 is not None:
                    X *= tw1[:, np.newaxis, np.newaxis, np.newaxis]
                if tw2 is not None:
                    X *= tw2[np.newaxis, :, np.newaxis, np.newaxis]
                if tw3 is not None:
                    X *= tw3[np.newaxis, np.newaxis, :, np.newaxis]

                K_hat = fftn(K, axes=(0, 1, 2))
                X_hat = fftn(X, axes=(0, 1, 2))
                X_hat *= K_hat[:, :, :, np.newaxis]
                X_inv = ifftn(X_hat, axes=(0, 1, 2))

                if tw1 is not None:
                    X_inv *= np.conj(tw1)[:, np.newaxis, np.newaxis, np.newaxis]
                if tw2 is not None:
                    X_inv *= np.conj(tw2)[np.newaxis, :, np.newaxis, np.newaxis]
                if tw3 is not None:
                    X_inv *= np.conj(tw3)[np.newaxis, np.newaxis, :, np.newaxis]

                accum += X_inv.real

    # scipy ifftn already divides by N, so just apply 1/2^d factor
    PhiGrid[:] = accum * 0.125


# ---- Normalization and force extraction ----

@numba.njit(cache=True)
def _zeta_and_force(Y, Phi, n, d):
    """Compute normalization Z and repulsive forces.

    Z = sum_i [(1 + 2*||Y_i||^2) * Phi[i,0] - 2 * Y_i . Phi[i,1:]] - n
    F[i,j] = (Y[i,j] * Phi[i,0] - Phi[i,j+1]) / Z
    """
    Z = 0.0
    for i in range(n):
        Ysq = 0.0
        for j in range(d):
            Ysq += Y[i, j] * Y[i, j]
            Z -= 2.0 * Y[i, j] * Phi[i, j + 1]
        Z += (1.0 + 2.0 * Ysq) * Phi[i, 0]
    Z -= n

    F = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        for j in range(d):
            F[i, j] = (Y[i, j] * Phi[i, 0] - Phi[i, j + 1]) / Z

    return F, Z


# ---- Top-level repulsive force computation ----

_S2G = {1: _s2g_1d, 2: _s2g_2d, 3: _s2g_3d}
_CONV = {1: _conv1d_nopad, 2: _conv2d_nopad, 3: _conv3d_nopad}
_G2S = {1: _g2s_1d, 2: _g2s_2d, 3: _g2s_3d}


def compute_repulsive_forces(
    Y: np.ndarray, h: float
) -> tuple[np.ndarray, float]:
    """Compute FFT-accelerated repulsive forces.

    Parameters
    ----------
    Y : ndarray of shape (n, d)
        Current embedding coordinates.
    h : float
        Grid side length for FFT.

    Returns
    -------
    Frep : ndarray of shape (n, d)
        Repulsive forces.
    zeta : float
        Normalization constant.
    """
    n, d = Y.shape
    eps = np.finfo(np.float64).eps

    # Guard against NaN/Inf input
    if not np.all(np.isfinite(Y)):
        return np.zeros((n, d), dtype=np.float64), 1.0

    # Work on copies — do NOT modify the embedding Y in-place
    Y_shifted = Y - Y.min(axis=0)

    # Global max across all dims
    maxy = Y_shifted.max()
    if maxy < eps:
        return np.zeros((n, d), dtype=np.float64), 1.0

    # Grid size
    n_grid = max(math.ceil(maxy / h), 14)
    n_grid = _best_grid_size(n_grid)

    # Grid positions in [0, n_grid-1]
    Y_grid = Y_shifted / maxy * (n_grid - 1)
    # Clamp to avoid edge issues
    Y_grid = np.clip(Y_grid, 0, n_grid - 1 - eps)

    # Exact grid spacing in data space
    h_exact = maxy / (n_grid - 1 - eps)

    ng = n_grid + 3  # padded grid: 4-point stencil at pos n_grid-1 needs idx n_grid+2
    n_vec = d + 1  # number of value channels

    # Setup scattered values: VScat[:, 0] = 1, VScat[:, 1:] = Y_shifted
    VScat = np.empty((n, n_vec), dtype=np.float64)
    VScat[:, 0] = 1.0
    VScat[:, 1:] = Y_shifted

    # Ensure grid coordinates are 2D for the numba kernels
    Y_grid_2d = np.ascontiguousarray(Y_grid.reshape(n, d))

    # S2G: scatter to grid
    grid_shape = tuple([ng] * d + [n_vec])
    VGrid = np.zeros(grid_shape, dtype=np.float64)
    _S2G[d](VGrid, Y_grid_2d, VScat, ng, n, n_vec)

    # G2G: FFT convolution
    PhiGrid = np.zeros(grid_shape, dtype=np.float64)
    _CONV[d](PhiGrid, VGrid, h_exact, ng, n_vec)

    # G2S: grid to scatter
    PhiScat = np.zeros((n, n_vec), dtype=np.float64)
    _G2S[d](PhiScat, PhiGrid, Y_grid_2d, ng, n, n_vec)

    # Compute Z and repulsive forces using original-scale coords
    Frep, zeta = _zeta_and_force(
        np.ascontiguousarray(Y_shifted), PhiScat, n, d
    )

    return Frep, zeta
