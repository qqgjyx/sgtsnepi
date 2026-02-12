"""Tests for Jaccard-index local weights."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from pysgtsnepi.utils.graph_weights import local_weights


def test_local_weights_triangle_pendant():
    """Verify Jaccard-index on a known graph: triangle (0-1-2) + pendant (3->0)."""
    # Adjacency (symmetric, binary):
    #   0 -- 1
    #   |  /       (no 2-3 edge)
    #   2    3--0
    #
    # Edges: 0-1, 0-2, 1-2, 0-3
    row = [0, 1, 0, 2, 1, 2, 0, 3]
    col = [1, 0, 2, 0, 2, 1, 3, 0]
    data = [1.0] * 8
    A = csc_matrix((data, (row, col)), shape=(4, 4))

    C = local_weights(A)

    eps = np.finfo(np.float64).eps

    # Degrees: d(0)=3, d(1)=2, d(2)=2, d(3)=1
    # Edge (0,1): N(0)={1,2,3}, N(1)={0,2} -> cap=|{2}|=1, denom=3+2-1=4 -> 0.25+eps
    assert C[0, 1] == pytest.approx(1 / 4 + eps)
    assert C[1, 0] == pytest.approx(1 / 4 + eps)

    # Edge (0,2): N(0)={1,2,3}, N(2)={0,1} -> cap=|{1}|=1, denom=3+2-1=4 -> 0.25+eps
    assert C[0, 2] == pytest.approx(1 / 4 + eps)

    # Edge (1,2): N(1)={0,2}, N(2)={0,1} -> cap=|{0}|=1, denom=2+2-1=3 -> 1/3+eps
    assert C[1, 2] == pytest.approx(1 / 3 + eps)

    # Edge (0,3): N(0)={1,2,3}, N(3)={0} -> cap=0, denom=3+1-0=4 -> 0+eps
    assert C[0, 3] == pytest.approx(0 + eps)
    assert C[3, 0] == pytest.approx(0 + eps)


def test_local_weights_preserves_sparsity():
    """Output should have same sparsity pattern as input."""
    row = [0, 1, 1, 2]
    col = [1, 0, 2, 1]
    A = csc_matrix(([1.0] * 4, (row, col)), shape=(3, 3))

    C = local_weights(A)

    assert C.nnz == A.nnz
    c_nz = sorted(zip(*C.nonzero(), strict=True))
    a_nz = sorted(zip(*A.nonzero(), strict=True))
    np.testing.assert_array_equal(c_nz, a_nz)


def test_local_weights_all_positive():
    """All weights should be strictly positive (eps floor)."""
    row = [0, 1, 0, 2, 1, 2, 0, 3]
    col = [1, 0, 2, 0, 2, 1, 3, 0]
    A = csc_matrix(([1.0] * 8, (row, col)), shape=(4, 4))

    C = local_weights(A)

    assert np.all(C.data > 0)
