# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-04-28

### Fixed
- Apply Jaccard-weight preprocessing to unweighted graphs before
  column-stochastic rescaling, so lambda rescaling is no longer degenerate
  on binary kNN inputs. Default `unweighted_to_weighted=True` in `api.py`
  and `estimator.py` matches the Julia reference's
  `flag_unweighted_to_weighted` (see `src/pysgtsnepi/utils/graph_weights.py`).

### Added
- Thin API surface for the auto-lambda and Lambda Lens helpers used by the
  IEEE VIS 2026 short paper (carried over from main since v0.3.0).

### Notes
- Julia-aligned default parameters introduced in v0.3.0 (commit 41d521d)
  remain unchanged. No breaking API changes since v0.3.0.

[0.3.1]: https://github.com/qqgjyx/sgtsnepi/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/qqgjyx/sgtsnepi/releases/tag/v0.3.0
