"""Sklearn-compatible estimator for SG-t-SNE-Pi."""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin

from pysgtsnepi.api import sgtsnepi
from pysgtsnepi.knn import build_knn_graph


class SGtSNEpi(BaseEstimator, TransformerMixin):
    """SG-t-SNE-Pi graph embedding.

    Parameters
    ----------
    d : int
        Embedding dimensions (default 2).
    lambda_ : float
        Rescaling parameter (default 10).
    n_neighbors : int
        Number of nearest neighbors for kNN (default 15).
    metric : str
        Distance metric (default "euclidean").
    max_iter : int
        Maximum iterations (default 1000).
    early_exag : int
        Early exaggeration iterations (default 250).
    alpha : float
        Exaggeration multiplier (default 12).
    eta : float
        Learning rate (default 200).
    h : float
        Grid side length for FFT. 0 = auto.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        d: int = 2,
        lambda_: float = 10.0,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        max_iter: int = 1000,
        early_exag: int = 250,
        alpha: float = 12.0,
        eta: float = 200.0,
        h: float = 0.0,
        random_state: int | None = None,
    ):
        self.d = d
        self.lambda_ = lambda_
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.max_iter = max_iter
        self.early_exag = early_exag
        self.alpha = alpha
        self.eta = eta
        self.h = h
        self.random_state = random_state

    def fit(self, X, y=None):
        """Compute the embedding.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            If dense, builds kNN graph first.
            If sparse, treats as adjacency matrix.

        Returns
        -------
        self
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Compute and return the embedding.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            If dense, builds kNN graph first.
            If sparse, treats as adjacency matrix.

        Returns
        -------
        Y : ndarray of shape (n_samples, d)
        """
        if issparse(X):
            A = X
        else:
            X = np.asarray(X, dtype=np.float64)
            A = build_knn_graph(
                X,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                random_state=self.random_state,
                perplexity=self.lambda_,
            )

        Y = sgtsnepi(
            A,
            d=self.d,
            lambda_=self.lambda_,
            max_iter=self.max_iter,
            early_exag=self.early_exag,
            alpha=self.alpha,
            eta=self.eta,
            h=self.h,
            random_state=self.random_state,
        )

        self.embedding_ = Y
        return Y
