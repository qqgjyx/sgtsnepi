"""kNN graph construction via PyNNDescent."""

from __future__ import annotations

import numpy as np
from pynndescent import NNDescent
from scipy.sparse import csc_matrix


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int | None = None,
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

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse (n_samples, n_samples) matrix of squared distances.
        Self-loops removed.
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

    return graph
