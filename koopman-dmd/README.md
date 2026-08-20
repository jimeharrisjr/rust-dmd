# koopman-dmd

[![Crates.io](https://img.shields.io/crates/v/koopman-dmd.svg)](https://crates.io/crates/koopman-dmd)
[![Documentation](https://docs.rs/koopman-dmd/badge.svg)](https://docs.rs/koopman-dmd)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Dynamic Mode Decomposition (DMD) with Koopman operator theory extensions, in pure Rust.

DMD extracts spatiotemporal coherent structures from time-series data, giving a linear
operator that approximates the dynamics of a possibly nonlinear system. This crate
implements the core method plus the Koopman-theoretic extensions used for analyzing
nonlinear and area-preserving systems.

Linear algebra is provided by [`faer`](https://crates.io/crates/faer); no external
BLAS/LAPACK installation is required.

## Features

- **Core DMD** — standard DMD with truncated SVD and optional mean centering
- **DMDc** — DMD with control (Proctor, Brunton & Kutz 2016): identifies `x_{t+1} = A x_t + B u_t` from explicit snapshot pairs, with a known-B variant and optional reduced-order output projection
- **Extended DMD** — polynomial, trigonometric, and delay-coordinate lifting for nonlinear systems
- **Hankel-DMD** — time-delay embedding via Krylov subspace, for scalar or low-dimensional signals
- **GLA** — Generalized Laplace Analysis for direct Koopman eigenfunction computation
- **Harmonic time averages** — phase space analysis and orbit classification via HTA
- **Mesochronic plots** — grid-based HTA visualization of mixed dynamics, parallelized with rayon
- **Built-in maps** — Chirikov standard, Froeschlé, extended standard, Hénon, and logistic
- **Prediction** — mode-based and matrix-based forecasting with lifting-aware back-projection
- **Analysis** — stability, spectrum, residuals, pseudospectrum, error metrics, dominant modes

## Installation

```toml
[dependencies]
koopman-dmd = "0.2"
```

Requires Rust 1.85 or later.

## Quick start

```rust
use koopman_dmd::{dmd, DmdConfig, dmd_spectrum, predict_modes};

// A 2-variable oscillating signal, as a 2 x 100 matrix
let n = 100;
let mut data = faer::Mat::<f64>::zeros(2, n);
for j in 0..n {
    let t = j as f64 * 0.1;
    data[(0, j)] = t.sin();
    data[(1, j)] = t.cos();
}

let result = dmd(&data, &DmdConfig::default()).unwrap();

// Inspect the spectrum
for m in &dmd_spectrum(&result, 0.1) {
    println!("freq={:.3} Hz, mag={:.3}, stability={:?}",
        m.frequency, m.magnitude, m.stability);
}

// Forecast 10 steps ahead
let pred = predict_modes(&result, 10, None).unwrap();
```

## DMD with control (DMDc)

When the system is driven by a known input, plain DMD folds the forcing into a biased
`A`. `dmdc` identifies the forced system `x_{t+1} = A x_t + B u_t` instead. It takes
explicit snapshot-pair matrices — `x1` (states at time `t`), `x2` (one step later), and
`u` (the input during each transition) — so columns may come from many concatenated
trajectories:

```rust
use koopman_dmd::{dmdc, DmdcConfig, stability_from_eigenvalues};

// Simulate x_{t+1} = A0 x_t + B0 u_t with
// A0 = [[0.9, 0.1], [0.0, 0.8]], B0 = [0.5, 1.0]
let m = 120;
let mut x1 = faer::Mat::<f64>::zeros(2, m);
let mut x2 = faer::Mat::<f64>::zeros(2, m);
let mut u = faer::Mat::<f64>::zeros(1, m);
let mut x = [1.0, -0.5];
for t in 0..m {
    // The input must be persistently exciting to identify A and B jointly
    let ut = (0.7 * t as f64).sin() + 0.5 * (2.3 * t as f64 + 1.0).cos();
    x1[(0, t)] = x[0];
    x1[(1, t)] = x[1];
    u[(0, t)] = ut;
    x = [0.9 * x[0] + 0.1 * x[1] + 0.5 * ut, 0.8 * x[1] + ut];
    x2[(0, t)] = x[0];
    x2[(1, t)] = x[1];
}

// Jointly recover A and B from the pairs
let config = DmdcConfig { rank_input: Some(3), ..Default::default() };
let result = dmdc(&x1, &x2, &u, &config).unwrap();
assert!((result.a[(0, 0)] - 0.9).abs() < 1e-8);
assert!((result.b[(1, 0)] - 1.0).abs() < 1e-8);

// Analyze the unforced dynamics via the eigenvalues of A
let stab = stability_from_eigenvalues(&result.eigenvalues, 1e-6);
assert!(stab.is_stable);
```

When the input coupling is known by construction, pin it with
`DmdcConfig { known_b: Some(b), .. }` and only `A` is estimated — this also avoids the
bias inherent in jointly fitting `A` and `B` from closed-loop (state-feedback) data.
With a zero-row `u`, `dmdc` performs autonomous multi-trajectory identification from
explicit pairs, which `dmd` cannot do. `DmdcConfig::rank_output` optionally projects
onto the leading SVD basis of `x2` for model reduction, and
`spectrum_from_eigenvalues` / `stability_from_eigenvalues` apply the standard analysis
tools to the identified spectrum.

## Extended DMD with lifting

Lifting maps observables into a higher-dimensional space where nonlinear dynamics become
approximately linear:

```rust
use koopman_dmd::{dmd, DmdConfig, LiftingConfig};

let n = 100;
let mut data = faer::Mat::<f64>::zeros(2, n);
for j in 0..n {
    let t = j as f64 * 0.05;
    data[(0, j)] = t.sin();
    data[(1, j)] = t.sin() * t.sin();
}

let config = DmdConfig {
    lifting: Some(LiftingConfig::Polynomial { degree: 2 }),
    ..Default::default()
};
let result = dmd(&data, &config).unwrap();
```

## Hankel-DMD

Time-delay embedding, for scalar signals or systems with limited measurements:

The signal is a `1 x n` matrix (one row per measured variable):

```rust
use koopman_dmd::{hankel_dmd, HankelConfig};

let n = 200;
let mut signal = faer::Mat::<f64>::zeros(1, n);
for j in 0..n {
    signal[(0, j)] = (j as f64 * 0.1).sin();
}

let config = HankelConfig { delays: Some(20), rank: Some(4), dt: 0.01 };
let result = hankel_dmd(&signal, &config).unwrap();
```

## Generalized Laplace Analysis

Direct computation of Koopman eigenfunctions via weighted time averages:

```rust
use koopman_dmd::{gla, GlaConfig};

let n = 200;
let mut data = faer::Mat::<f64>::zeros(2, n);
for j in 0..n {
    let t = j as f64 * 0.05;
    data[(0, j)] = t.sin();
    data[(1, j)] = t.cos();
}

let config = GlaConfig {
    eigenvalues: None,   // auto-detect
    n_eigenvalues: 4,
    tol: 1e-6,
    max_iter: None,
};
let result = gla(&data, &config).unwrap();
```

## Harmonic time averages and mesochronic plots

Phase space analysis of area-preserving maps:

```rust
use koopman_dmd::{harmonic_time_average, mesochronic_compute, Observable, StandardMap};

let map = StandardMap { epsilon: 0.12 };

// HTA at a single initial condition
let hta = harmonic_time_average(&[0.5, 0.3], &map, &Observable::SinPi, 0.5, 1000).unwrap();

// Mesochronic plot over a grid (parallelized with rayon)
let mhp = mesochronic_compute(
    &map, (0.0, 1.0), (0.0, 1.0), 32, &Observable::SinPi, 0.5, 1000,
).unwrap();
```

## Other languages

Python and R bindings live in the same repository:

- **Python** — [`koopman-dmd` on PyPI](https://pypi.org/project/koopman-dmd/) (PyO3 + maturin)
- **R** — `koopman.dmd` (extendr)

## Benchmarks

```bash
cargo bench
```

Representative results (Apple Silicon):

| Operation | Size | Time |
|-----------|------|------|
| DMD | 5 × 100 | ~80 µs |
| DMD | 50 × 1000 | ~8.5 ms |
| Predict (modes) | 2 vars, 100 steps | ~215 µs |
| Predict (matrix) | 2 vars, 100 steps | ~36 µs |
| Hankel-DMD | 1 × 200, 20 delays | ~130 µs |
| GLA | 2 × 200 | ~344 µs |

## References

- Schmid, P.J. (2010). Dynamic mode decomposition of numerical and experimental data.
  *Journal of Fluid Mechanics*, 656, 5–28. [doi:10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217)
- Kutz, J.N., Brunton, S.L., Brunton, B.W., & Proctor, J.L. (2016).
  *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.
  [doi:10.1137/1.9781611974508](https://doi.org/10.1137/1.9781611974508)
- Mezić, I. (2020). Spectrum of the Koopman operator, spectral expansions in functional
  spaces, and state-space geometry. [arXiv:2009.05883](https://arxiv.org/abs/2009.05883)
- Levnajić, Z. & Mezić, I. (2014). Ergodic theory and visualization.
  [arXiv:0808.2182v2](https://arxiv.org/abs/0808.2182)

## License

MIT — see [LICENSE](LICENSE).
