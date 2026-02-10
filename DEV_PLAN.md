# PySGtSNEpi — Development Plan

> Living doc. Update as work progresses.

## Status: scaffolding complete, starting core implementation

### What's Done

| Item | Status |
|------|--------|
| Repo scaffolding (src layout, pyproject.toml, CI, docs) | Done |
| PyPI name claimed (`pysgtsnepi`) | Done |
| README + docs/index.md (doubles as spec) | Done |
| Lambda equalization (`utils/sgtsne_lambda_equalization.py`) | Done |
| Repo cleanup (removed placeholders, MathJax, binary data, etc.) | Done |

### What's Next — Implementation Order

Work through the README roadmap top-to-bottom. Each item maps to a
module under `src/pysgtsnepi/`. Primary translation source is the
**Julia reference** (`ref/SGtSNEpi.jl/`), cross-checked against the
C++ reference (`ref/sgtsnepi/`).

#### Phase 1: Core embedding loop

1. **kNN graph construction** — `src/pysgtsnepi/knn.py`
   - Wrap PyNNDescent to build a sparse kNN graph from a point cloud
   - Convert to CSR stochastic matrix (row-normalize)
   - Julia ref: `SGtSNEpi.jl/src/knn.jl`

2. **Attractive forces** — `src/pysgtsnepi/attractive.py`
   - Sparse matrix–vector product for attractive gradient
   - Julia ref: `SGtSNEpi.jl/src/attractive.jl`

3. **Repulsive forces (direct)** — `src/pysgtsnepi/repulsive.py`
   - Grid interpolation + FFT convolution (polynomial kernel)
   - Numba JIT for interpolation scatter/gather
   - Julia ref: `SGtSNEpi.jl/src/repulsive.jl`

4. **Embedding loop** — `src/pysgtsnepi/embedding.py`
   - Adam or momentum SGD optimizer
   - Early exaggeration schedule
   - Combines attractive + repulsive gradients
   - Julia ref: `SGtSNEpi.jl/src/sgtsnepi.jl`

#### Phase 2: Public API

5. **Functional API** — `src/pysgtsnepi/api.py`
   - `sgtsnepi(P, d=2, lambda_=1, ...)` — takes sparse graph, returns embedding
   - Orchestrates lambda equalization → embedding loop

6. **Sklearn estimator** — `src/pysgtsnepi/estimator.py`
   - `SGtSNEpi(d=2, lambda_=10, n_neighbors=15, ...)` — `BaseEstimator` + `TransformerMixin`
   - `fit_transform(X)`: kNN → stochastic graph → sgtsnepi()
   - `fit_transform(P)` when input is already sparse

7. **Wire up `__init__.py`** — export `SGtSNEpi` and `sgtsnepi`

#### Phase 3: Polish

8. **Tests** — `tests/`
   - Unit tests for each module (knn, attractive, repulsive, embedding)
   - Integration test: small synthetic graph → embedding → verify shape
   - Regression test: known output on a fixed seed

9. **1D / 3D support** — generalize `d` parameter through the pipeline

10. **Docs & examples** — update docs/examples/ with a real end-to-end notebook

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
