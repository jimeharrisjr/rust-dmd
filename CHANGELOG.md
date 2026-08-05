# Changelog

All notable changes to the `koopman-dmd` Rust crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Python (`koopman-dmd` on PyPI) and R (`koopman.dmd`) packages version independently;
see their respective changelogs.

## [Unreleased]

## [0.1.0] - 2026-08-05

Initial release.

### Added

- **Core DMD** — standard Dynamic Mode Decomposition with truncated SVD and optional mean centering
- **Extended DMD** — polynomial, polynomial-with-cross-terms, trigonometric, and
  delay-coordinate lifting for nonlinear systems
- **Hankel-DMD** — time-delay embedding via Krylov subspace
- **GLA** — Generalized Laplace Analysis for direct Koopman eigenfunction computation
- **Harmonic time averages** — phase space analysis and orbit classification
- **Mesochronic plots** — grid-based HTA visualization, parallelized with rayon
- **Built-in maps** — Chirikov standard, Froeschlé, extended standard, Hénon, and logistic
- **Prediction** — mode-based and matrix-based forecasting with lifting-aware back-projection
- **Analysis** — stability, spectrum, residuals, pseudospectrum, error metrics, dominant modes

[Unreleased]: https://github.com/jimeharrisjr/rust-dmd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jimeharrisjr/rust-dmd/releases/tag/v0.1.0
