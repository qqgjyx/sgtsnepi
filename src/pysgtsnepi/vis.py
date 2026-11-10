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
        Class labels for coloring, treated as categories (any values that
        ``np.unique`` can sort). If *None*, all points get the same color.
    A : sparse matrix, optional
        Adjacency matrix. If provided, edges are drawn on the plot.
    ax : matplotlib Axes, optional
        Axes to draw on. If *None*, a new figure is created. For 3D embeddings
        a 3D projection Axes is created automatically.
    cmap : str
        Matplotlib colormap name. Each class gets its own color: a discrete
        colormap is used in order, and if it has fewer colors than there are
        classes, ``"turbo"`` is sampled instead (distinct for up to 256
        classes).
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
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if Y.shape[1] not in (2, 3):
        raise ValueError("show_embedding only supports 2D or 3D embeddings")

    n = Y.shape[0]
    if L is None:
        L = np.zeros(n, dtype=int)
    # Map labels to class ids 0..n_classes-1 so colors depend only on class
    labels_unique, L = np.unique(np.asarray(L), return_inverse=True)
    n_classes = len(labels_unique)

    # One RGBA color per class
    cm = plt.get_cmap(cmap)
    if isinstance(cm, ListedColormap) and cm.N >= n_classes:
        palette = cm(np.arange(n_classes))
    else:
        if isinstance(cm, ListedColormap):
            cm = plt.get_cmap("turbo")
        palette = cm(np.linspace(0, 1, n_classes))

    is_3d = Y.shape[1] == 3
    Collection = Line3DCollection if is_3d else LineCollection

    if ax is None:
        if is_3d:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        if is_3d != (getattr(ax, "name", None) == "3d"):
            raise ValueError(
                f"{Y.shape[1]}D embedding needs a {'3D' if is_3d else '2D'} Axes"
            )
        fig = ax.get_figure()

    # --- Draw edges ---
    if A is not None and issparse(A):
        A_lower = tril(A)
        rows, cols = A_lower.nonzero()

        L_rows = L[rows]
        L_cols = L[cols]

        # Intra-cluster edges (grouped by class for correct coloring)
        for k in range(n_classes):
            mask = (L_rows == k) & (L_cols == k)
            if mask.any():
                segs = np.stack([Y[rows[mask]], Y[cols[mask]]], axis=1)  # (m, 2, d)
                color = [*palette[k, :3], edge_alpha]
                ax.add_collection(Collection(segs, colors=[color], linewidths=lwd_in))

        # Inter-cluster edges (gray)
        mask_cross = L_rows != L_cols
        if mask_cross.any():
            segs = np.stack([Y[rows[mask_cross]], Y[cols[mask_cross]]], axis=1)
            ax.add_collection(
                Collection(
                    segs,
                    colors=[(0.67, 0.73, 0.73, edge_alpha)],
                    linewidths=lwd_out,
                )
            )

    # --- Draw nodes ---
    ax.scatter(*Y.T, c=palette[L], s=mrk_size, zorder=2)

    if not is_3d:
        ax.set_aspect("equal")
    ax.autoscale_view()

    # Legend (only if multiple classes)
    if n_classes > 1:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette[k],
                markersize=8,
                label=str(label),
            )
            for k, label in enumerate(labels_unique)
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=min(n_classes, 10),
            frameon=False,
        )

    return fig, ax
