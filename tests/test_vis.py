"""Tests for show_embedding."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random

from pysgtsnepi.vis import show_embedding


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _plot(d, L, A=None):
    rng = np.random.default_rng(0)
    return show_embedding(rng.standard_normal((len(L), d)), L, A=A)


def _node_colors(ax):
    return ax.collections[-1].get_facecolors()


def _n_distinct(colors):
    return len(np.unique(np.round(colors[:, :3], 6), axis=0))


@pytest.mark.parametrize("d", [2, 3])
def test_renders_2d_and_3d(d):
    n = 50
    L = np.random.default_rng(1).integers(0, 4, n)
    A = sparse_random(n, n, density=0.1, format="csr", random_state=0)
    fig, ax = _plot(d, L, A=A + A.T)

    assert (ax.name == "3d") == (d == 3)
    edge_type = Line3DCollection if d == 3 else LineCollection
    assert all(isinstance(c, edge_type) for c in ax.collections[:-1])
    fig.canvas.draw()


@pytest.mark.parametrize(
    "labels",
    [[0, 5, 9], [0.0, 5.0, 9.0], [3, 100, 7], ["b", "a", "c"]],
)
def test_nodes_edges_legend_share_colors(labels):
    L = np.repeat(labels, 2)  # two nodes per class, joined by an edge
    n = len(L)
    rows = np.arange(1, n, 2)
    A = csr_matrix((np.ones(len(rows)), (rows, rows - 1)), shape=(n, n))
    _, ax = _plot(2, L, A=A)

    nodes = _node_colors(ax)
    assert _n_distinct(nodes) == len(labels)

    leg = ax.get_legend()
    legend = {
        t.get_text(): h.get_markerfacecolor()
        for t, h in zip(leg.get_texts(), leg.get_lines(), strict=True)
    }
    assert sorted(legend) == sorted(str(x) for x in labels)

    for edges in ax.collections[:-1]:
        rgb = edges.get_colors()[0][:3]
        k = np.flatnonzero(np.all(np.isclose(nodes[:, :3], rgb), axis=1))
        assert k.size == 2  # matches exactly one class (its two nodes)
        np.testing.assert_allclose(legend[str(L[k[0]])][:3], rgb)


@pytest.mark.parametrize("cmap", ["tab10", "viridis"])
def test_many_classes_get_distinct_colors(cmap):
    L = np.arange(12)
    _, ax = _plot(2, L)
    assert _n_distinct(_node_colors(ax)) == 12


@pytest.mark.parametrize("L", [None, np.zeros(5, dtype=int)])
def test_single_class_has_no_legend(L):
    _, ax = show_embedding(np.zeros((5, 2)), L)
    assert ax.get_legend() is None


@pytest.mark.parametrize("d, projection", [(3, None), (2, "3d")])
def test_rejects_mismatched_axes(d, projection):
    fig = plt.figure()
    ax = fig.add_subplot(projection=projection)
    with pytest.raises(ValueError, match="Axes"):
        show_embedding(np.zeros((5, d)), ax=ax)


def test_rejects_bad_dim():
    with pytest.raises(ValueError, match="2D or 3D"):
        show_embedding(np.zeros((5, 4)))
