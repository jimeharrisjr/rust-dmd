# Changelog

Notable changes to the `koopman-dmd` Python package.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
It versions independently of the [Rust crate](../CHANGELOG.md) and the R package.

## [Unreleased]

## [0.1.0] - 2026-08-05

Initial release.

### Added

- `DMD` — core Dynamic Mode Decomposition, with optional Extended DMD lifting
  (`polynomial`, `polynomial_cross`, `trigonometric`, `delay`)
- `HankelDMD` — time-delay embedding for scalar or low-dimensional signals
- `GLA` — Generalized Laplace Analysis for Koopman eigenfunctions
- `generate_trajectory`, `harmonic_time_average`, `hta_convergence`,
  `mesochronic_compute`, `classify_phase_space` — phase space analysis for the
  Chirikov standard, Froeschlé, extended standard, Hénon, and logistic maps
- Prebuilt `abi3` wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and
  Windows (x86_64), covering Python 3.9 and later with a single wheel per platform

[Unreleased]: https://github.com/jimeharrisjr/rust-dmd/compare/py-v0.1.0...HEAD
[0.1.0]: https://github.com/jimeharrisjr/rust-dmd/releases/tag/py-v0.1.0
