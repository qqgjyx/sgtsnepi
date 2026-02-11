# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PySGtSNEpi is a pure Python implementation of the SG-t-SNE-Π graph embedding algorithm (no C++ extensions). It translates primarily from the Julia reference implementation (`ref/SGtSNEpi.jl/`), cross-checked against the C++ original (`ref/sgtsnepi/`). The goal is `pip install pysgtsnepi` working on all platforms.

## Commands

```bash
# Environment (uses uv, not pip/conda)
uv sync                          # Install core deps
uv sync --extra dev              # Install with dev tools (pytest, ruff, pre-commit)
uv sync --extra docs             # Install with docs tools (mkdocs)

# Testing
uv run pytest                    # Run all tests
uv run pytest --cov              # Run tests with coverage
uv run pytest tests/test_foo.py  # Run a single test file
uv run pytest -k "test_name"     # Run a specific test by name

# Linting & formatting
uv run ruff check src/ tests/    # Lint
uv run ruff check --fix src/     # Lint with autofix
uv run ruff format src/ tests/   # Format

# Docs
uv run mkdocs serve              # Local docs preview
```

## Architecture

**src layout** — all source lives under `src/pysgtsnepi/`.

The implementation follows a pipeline: raw data → kNN graph (+ perplexity equalization) → column-stochastic matrix → lambda rescaling → symmetrize → embedding loop (attractive + repulsive forces) → low-dimensional coordinates.

### Module map

| Module | Purpose | Reference source |
|--------|---------|-----------------|
| `knn.py` | Build sparse kNN graph via PyNNDescent, optional perplexity equalization | `ref/SGtSNEpi.jl/src/knn.jl`, `util.jl` |
| `attractive.py` | Sparse matrix–vector attractive gradient | `ref/SGtSNEpi.jl/src/attractive.jl` |
| `repulsive.py` | FFT-accelerated repulsive forces (grid interpolation + convolution) | `ref/SGtSNEpi.jl/src/repulsive.jl` |
| `embedding.py` | Optimizer loop (Adam/momentum SGD, early exaggeration) | `ref/SGtSNEpi.jl/src/sgtsnepi.jl` |
| `api.py` | Functional API: `sgtsnepi(P, d=2, ...)` | — |
| `estimator.py` | sklearn-compatible `SGtSNEpi` estimator | — |
| `utils/graph_rescaling.py` | Lambda-based graph rescaling | `ref/sgtsnepi/src/graph_rescaling.cpp` |

### Key dependencies

- **numpy/scipy**: core math, sparse matrices, FFT
- **numba**: JIT for interpolation kernels and gradient accumulation
- **pynndescent**: approximate nearest neighbor search
- **scikit-learn**: `BaseEstimator`/`TransformerMixin` for API compatibility

All four (numpy, scipy, numba, pynndescent, scikit-learn) are core dependencies, always installed.

## Conventions

- **Python ≥ 3.10**, version `0.3.0`
- **Ruff** for linting+formatting: line length 88, rules E/F/W/B/I/UP/RUF
- **pre-commit** hooks: trailing whitespace, EOF fixer, YAML/TOML checks, large file guard (500KB), ruff
- CI runs lint → test (matrix: ubuntu/macOS/Windows × Python 3.10/3.12) → docs deploy
- Docs use **mkdocs-material** with **KaTeX** (not MathJax), **awesome-pages** for navigation
- `uv.lock` is committed for reproducibility
