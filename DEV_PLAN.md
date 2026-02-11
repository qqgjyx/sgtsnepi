# PySGtSNEpi — Development Plan

> Living doc. Update as work progresses.

## Status: Phase 1+2 complete, entering polish phase

### What's Done

| Item | Status |
|------|--------|
| Repo scaffolding (src layout, pyproject.toml, CI, docs) | Done |
| PyPI name claimed (`pysgtsnepi`) | Done |
| README + docs/index.md (doubles as spec) | Done |
| Lambda equalization (`utils/sgtsne_lambda_equalization.py`) | Done |
| Repo cleanup (removed placeholders, MathJax, binary data, etc.) | Done |
| kNN graph construction (`knn.py`) | Done |
| Attractive forces (`attractive.py`) | Done |
| Repulsive forces — FFT-accelerated (`repulsive.py`) | Done |
| Embedding loop (`embedding.py`) | Done |
| Functional API (`api.py`) | Done |
| Sklearn estimator (`estimator.py`) | Done |
| `__init__.py` exports wired up | Done |
| Tests (23 tests passing) | Done |
| 1D / 3D embedding support | Done |
| Post-cleanup (bisection fix, vectorized kernels, API tightening) | Done |
| API reference docs for all modules | Done |
| Roadmaps updated (README, docs/index.md) | Done |

### What's Next

#### Phase 3: Polish (in progress)

1. **Examples & tutorials** — end-to-end notebook with real dataset
2. **Performance benchmarking** — compare with Julia/C++ implementations
3. **CI pipeline verification** — confirm matrix builds pass on all platforms

### Reference Map

| Module | Julia source | C++ source |
|--------|-------------|------------|
| knn | `ref/SGtSNEpi.jl/src/knn.jl` | N/A (uses FLANN) |
| attractive | `ref/SGtSNEpi.jl/src/attractive.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |
| repulsive | `ref/SGtSNEpi.jl/src/repulsive.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |
| embedding | `ref/SGtSNEpi.jl/src/sgtsnepi.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |
| lambda eq | `ref/SGtSNEpi.jl/src/lambda.jl` | `ref/sgtsnepi/src/sgtsne.cpp` |

### Architecture Decisions

- **Pure Python + NumPy/SciPy + Numba** — no C++ extension modules
- **PyNNDescent** for kNN (pure Python, supports many metrics)
- **Numba** for interpolation kernels and gradient accumulation
- **scipy.fft** for FFT convolution (repulsive forces)
- **sklearn BaseEstimator** for API compatibility

### Dev Environment

```bash
uv sync              # install deps
uv run pytest        # run tests
uv run mkdocs serve  # preview docs
```
