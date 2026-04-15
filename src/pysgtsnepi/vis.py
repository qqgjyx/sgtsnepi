"""Embedding visualization utilities.

Translates ``vis.jl`` from the Julia SGtSNEpi.jl reference using matplotlib.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse, tril


def show_embedding(
    Y: np.ndarray,
    L: np.ndarray | None = None,
    *,
    A=None,
    ax=None,
    cmap: str = "tab10",
    edge_alpha: float = 0.2,
    lwd_in: float = 0.5,
    lwd_out: float = 0.3,
    mrk_size: float = 4,
    figsize: tuple[int, int] = (10, 10),
    dpi: int = 150,
):
    """Visualize a 2D or 3D embedding with optional graph edges.

    Matches Julia ``SGtSNEpi.jl`` ``show_embedding``.

    Parameters
    ----------
    Y : ndarray of shape (n, 2) or (n, 3)
        2D or 3D embedding coordinates.
    L : ndarray of shape (n,), optional
        Integer class labels for coloring. If *None*, all points get the
        same color.
    A : sparse matrix, optional
        Adjacency matrix. If provided, edges are drawn on the plot.
    ax : matplotlib Axes, optional
        Axes to draw on. If *None*, a new figure is created. For 3D embeddings
        a 3D projection Axes is created automatically.
    cmap : str
        Matplotlib colormap name.
    edge_alpha : float
        Alpha channel for edge lines.
    lwd_in : float
        Line width for intra-cluster edges.
    lwd_out : float
        Line width for inter-cluster edges.
    mrk_size : float
        Marker size for scatter points.
    figsize : tuple
        Figure size if *ax* is None.
    dpi : int
        Figure DPI if *ax* is None (default 150).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes (3D Axes for 3D embeddings)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if Y.shape[1] not in (2, 3):
        raise ValueError("show_embedding only supports 2D or 3D embeddings")

    n = Y.shape[0]
    if L is None:
        L = np.zeros(n, dtype=int)
    L = np.asarray(L)

    # Shift labels to start at 0
    L = L - L.min()
    labels_unique = np.unique(L)
    n_classes = len(labels_unique)

    # Get colormap colors
    cm = plt.get_cmap(cmap, max(n_classes, 10))

    is_3d = Y.shape[1] == 3

    if ax is None:
        if is_3d:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    # --- Draw edges ---
    if A is not None and issparse(A):
        A_lower = tril(A)
        rows, cols = A_lower.nonzero()

        L_rows = L[rows]
        L_cols = L[cols]

        # Intra-cluster edges (grouped by class for correct coloring)
        for k in labels_unique:
            mask = (L_rows == k) & (L_cols == k)
            if mask.any():
                segs = np.stack(
                    [Y[rows[mask]], Y[cols[mask]]], axis=1
                )  # (m, 2, 2) or (m, 2, 3)
                color = [*cm(k)[:3], edge_alpha]
                if is_3d:
                    lc = Line3DCollection(segs, colors=[color], linewidths=lwd_in)
                else:
                    lc = LineCollection(segs, colors=[color], linewidths=lwd_in)
                ax.add_collection(lc)

        # Inter-cluster edges (gray)
        mask_cross = L_rows != L_cols
        if mask_cross.any():
            segs = np.stack([Y[rows[mask_cross]], Y[cols[mask_cross]]], axis=1)
            if is_3d:
                lc = Line3DCollection(
                    segs,
                    colors=[(0.67, 0.73, 0.73, edge_alpha)],
                    linewidths=lwd_out,
                )
            else:
                lc = LineCollection(
                    segs,
                    colors=[(0.67, 0.73, 0.73, edge_alpha)],
                    linewidths=lwd_out,
                )
            ax.add_collection(lc)

    # --- Draw nodes ---
    if is_3d:
        sc = ax.scatter(
            Y[:, 0],
            Y[:, 1],
            Y[:, 2],
            c=L,
            cmap=cmap,
            s=mrk_size,
            zorder=2,
            vmin=labels_unique.min(),
            vmax=labels_unique.max(),
        )
    else:
        sc = ax.scatter(
            Y[:, 0],
            Y[:, 1],
            c=L,
            cmap=cmap,
            s=mrk_size,
            zorder=2,
            vmin=labels_unique.min(),
            vmax=labels_unique.max(),
        )

    if not is_3d:
        ax.set_aspect("equal")
    ax.autoscale_view()

    # Colorbar on the right for both 2D and 3D
    if n_classes > 1:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("SOFA Score Cluster", fontsize=10)

    return fig, ax
