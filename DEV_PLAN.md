# PySGtSNEpi — Development Plan

> Living doc. Update as work progresses.

## Status: Phase 1+2+3+3b complete, v0.3.0

### What's Done

| Item | Status |
|------|--------|
| Repo scaffolding (src layout, pyproject.toml, CI, docs) | Done |
| PyPI name claimed (`pysgtsnepi`) | Done |
| README + docs/index.md (doubles as spec) | Done |
| Graph rescaling (`utils/graph_rescaling.py`, matches C++ `lambdaRescaling`) | Done |
| Perplexity equalization (in `knn.py`, matches Julia `perplexity_equalization`) | Done |
| Repo cleanup (removed placeholders, MathJax, binary data, etc.) | Done |
| kNN graph construction (`knn.py`) | Done |
| Attractive forces (`attractive.py`) | Done |
| Repulsive forces — FFT-accelerated (`repulsive.py`) | Done |
| Embedding loop (`embedding.py`) | Done |
| Functional API (`api.py`) | Done |
| Sklearn estimator (`estimator.py`) | Done |
| `__init__.py` exports wired up | Done |
| Tests (25 tests passing, 0 warnings) | Done |
| 1D / 3D embedding support | Done |
| Post-cleanup (bisection fix, vectorized kernels, API tightening) | Done |
| API reference docs for all modules | Done |
| Roadmaps updated (README, docs/index.md) | Done |
| Embedding quality fix (perplexity eq + lambda rescaling + HOG fix) | Done |
| Examples: digits quickstart + MNIST embedding notebooks | Done |
| `vis.py` — `show_embedding` visualization helper | Done |
| `graph_weights.py` — Jaccard-index local weights for unweighted graphs | Done |
| `03_graph_embedding.ipynb` — optdigits_10NN graph embedding tutorial | Done |
| Julia-aligned defaults (h=1.0, Y0 scale 0.01, vis params) | Done |
| Docs nav updated (`docs/examples/.pages`) | Done |
| Tests (29 tests passing, 0 warnings) | Done |

### What's Next

#### Phase 4: Optimization & Release

1. **Performance benchmarking** — compare with Julia/C++ implementations
2. **CI pipeline verification** — confirm matrix builds pass on all platforms
3. **Numba parallelization** — `prange` for perplexity eq and lambda rescaling loops

### Reference Map

| Module | Julia source | C++ source |
|--------|-------------|------------|
| knn | `ref/SGtSNEpi.jl/src/knn.jl` | N/A (uses FLANN) |
| perplexity eq | `ref/SGtSNEpi.jl/src/util.jl` | N/A |
| attractive | `ref/SGtSNEpi.jl/src/attractive.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |
| repulsive | `ref/SGtSNEpi.jl/src/repulsive.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |
| embedding | `ref/SGtSNEpi.jl/src/sgtsnepi.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |
| graph rescaling | `ref/SGtSNEpi.jl/src/util.jl` | `ref/sgtsnepi/src/graph_rescaling.cpp` |

### Architecture Decisions

- **Pure Python + NumPy/SciPy + Numba** — no C++ extension modules
- **PyNNDescent** for kNN (pure Python, supports many metrics)
- **Numba** for interpolation kernels and gradient accumulation
- **scipy.fft** for FFT convolution (repulsive forces)
- **scipy.special.logsumexp** for numerically stable bisection in graph rescaling
- **sklearn BaseEstimator** for API compatibility

### Dev Environment

```bash
uv sync              # install deps
uv run pytest        # run tests
uv run mkdocs serve  # preview docs
```
