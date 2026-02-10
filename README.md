# <img src="docs/assets/sgtsne.png" width="40px" align="center" alt="SG-t-SNE-Pi logo"> PySGtSNEpi

<img src="docs/assets/logo.png" width="800px" align="center" alt="SG-t-SNE-Pi embedding demo">

[![PyPI version](https://badge.fury.io/py/pysgtsnepi.svg)](https://pypi.org/project/pysgtsnepi/)
[![Python 3.10–3.13](https://img.shields.io/pypi/pyversions/pysgtsnepi)](https://pypi.org/project/pysgtsnepi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/qqgjyx/sgtsnepi/actions/workflows/ci.yml/badge.svg)](https://github.com/qqgjyx/sgtsnepi/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-qqgjyx.com%2Fpysgtsnepi-blue)](https://qqgjyx.com/pysgtsnepi/)

> Embed sparse graphs into 2D/3D — pure Python, pip-installable, sklearn-compatible.

PySGtSNEpi is a pure Python port of the
[SG-t-SNE-Pi](https://t-sne-pi.cs.duke.edu) algorithm, translated from the
original [C++](https://github.com/fcdimitr/sgtsnepi) and
[Julia](https://github.com/fcdimitr/SGtSNEpi.jl) implementations.
Unlike standard t-SNE, SG-t-SNE-Pi works on **any sparse stochastic graph**,
not just kNN graphs derived from point clouds.
No C/C++ compiler needed — `pip install` and go.

## Features

- **1D / 2D / 3D embedding** of sparse stochastic graphs
- **Arbitrary sparse graph input** — not limited to kNN graphs
- **Point cloud input** with automatic kNN graph construction via [PyNNDescent](https://github.com/lmcinnes/pynndescent)
- **Lambda rescaling** to equalize effective node degrees
- **Scikit-learn compatible** API (`fit` / `transform` / `fit_transform`)
- **Pure Python** — runs on Windows, macOS (including Apple Silicon), and Linux
- **Numba JIT** compiled hot loops for near-native speed
- **FFT-accelerated** repulsive force computation

## Quick Start

```bash
pip install pysgtsnepi
```

### Scikit-learn API (point cloud)

```python
from pysgtsnepi import SGtSNEpi

model = SGtSNEpi(d=2, lambda_=10)
Y = model.fit_transform(X)   # X is (n_samples, n_features)
```

### Functional API (sparse graph)

```python
from scipy.io import mmread
from pysgtsnepi import sgtsnepi

P = mmread("graph.mtx")       # sparse stochastic graph
Y = sgtsnepi(P, d=3, lambda_=10)
```

## Roadmap

- [x] Lambda equalization
- [ ] kNN graph construction (via PyNNDescent)
- [ ] Core SG-t-SNE-Pi embedding (attractive + repulsive forces)
- [ ] FFT-accelerated repulsive forces
- [ ] Numba JIT for interpolation and gradient kernels
- [ ] 1D / 3D embedding support
- [ ] `SGtSNEpi` sklearn estimator class
- [ ] `sgtsnepi()` functional API

## Citation

If you use this package in your research, please cite:

```bibtex
@article{pitsianis2019joss,
  title     = {{SG-t-SNE-$\Pi$}: Swift Neighbor Embedding of Sparse Stochastic Graphs},
  author    = {Pitsianis, Nikos and Floros, Dimitris and Iliopoulos, Alexandros-Stavros and Sun, Xiaobai},
  journal   = {Journal of Open Source Software},
  volume    = {4},
  number    = {39},
  pages     = {1577},
  year      = {2019},
  doi       = {10.21105/joss.01577}
}

@inproceedings{pitsianis2019hpec,
  title     = {Spaceland Embedding of Sparse Stochastic Graphs},
  author    = {Pitsianis, Nikos and Iliopoulos, Alexandros-Stavros and Floros, Dimitris and Sun, Xiaobai},
  booktitle = {IEEE High Performance Extreme Computing Conference},
  year      = {2019},
  doi       = {10.1109/HPEC.2019.8916505}
}
```

## Links

- [Algorithm website](https://t-sne-pi.cs.duke.edu)
- [C++ implementation](https://github.com/fcdimitr/sgtsnepi)
- [Julia implementation](https://github.com/fcdimitr/SGtSNEpi.jl)
- [Documentation](https://qqgjyx.com/pysgtsnepi/)

## License

MIT — see [LICENSE](https://github.com/qqgjyx/sgtsnepi/blob/main/LICENSE).
