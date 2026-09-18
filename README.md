# koopman-dmd

Dynamic Mode Decomposition with Koopman operator theory extensions, implemented in Rust with Python and R bindings.

## Features

- **Core DMD** -- Standard Dynamic Mode Decomposition with truncated SVD and optional mean centering
- **DMDc** -- DMD with control (Proctor, Brunton & Kutz 2016): identifies `x_{t+1} = A x_t + B u_t` from explicit snapshot pairs, with a known-B variant and optional reduced-order output projection
- **Extended DMD** -- Polynomial, trigonometric, and delay-coordinate lifting for nonlinear systems
- **Hankel-DMD** -- Time-delay embedding via Krylov subspace for scalar or low-dimensional signals
- **GLA** -- Generalized Laplace Analysis for direct eigenfunction computation
- **Harmonic Time Averages** -- Phase space analysis and orbit classification via HTA
- **Mesochronic Harmonic Plots** -- Parallelized grid-based HTA visualization of mixed dynamics
- **Built-in maps** -- Chirikov standard map, Froeschle, extended standard, Henon, and logistic maps
- **Prediction** -- Mode-based and matrix-based forecasting with lifting-aware back-projection
- **Analysis** -- Stability, spectrum, residuals, pseudospectrum, error metrics, dominant mode extraction

## Project Structure

```
koopman-dmd/          Rust core library (crate)
koopman-dmd-py/       Python bindings (PyO3 + maturin)
koopman-dmd-r/        R package (extendr)
```

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
koopman-dmd = "0.2"
```

Requires Rust 1.85 or later.

### Python

```bash
pip install koopman-dmd
```

Prebuilt wheels for Linux, macOS, and Windows on Python 3.9+. To build from a checkout
instead, requires a Rust toolchain:

```bash
pip install maturin
cd koopman-dmd-py
maturin develop --release
```

### R

From CRAN:

```r
install.packages("koopman.dmd")
```

On Windows and macOS, CRAN provides prebuilt binaries. Installing from source
(Linux, or `type = "source"` elsewhere) compiles the bundled Rust sources
locally and requires a Rust toolchain (`rustc` >= 1.85). To install from a
checkout of this repository:

```bash
R CMD INSTALL koopman-dmd-r
```

## Quick Start

### Rust

```rust
use koopman_dmd::{dmd, DmdConfig, predict_modes, dmd_spectrum};

// Create a 2-variable oscillating signal (2 x 100 matrix)
let n = 100;
let mut data = faer::Mat::<f64>::zeros(2, n);
for j in 0..n {
    let t = j as f64 * 0.1;
    data[(0, j)] = t.sin();
    data[(1, j)] = t.cos();
}

// Compute DMD
let config = DmdConfig::default();
let result = dmd(&data, &config).unwrap();

// Inspect spectrum
let modes = dmd_spectrum(&result, 0.1);
for m in &modes {
    println!("freq={:.3} Hz, mag={:.3}, stability={:?}",
        m.frequency, m.magnitude, m.stability);
}

// Predict 10 steps ahead
let pred = predict_modes(&result, 10, None).unwrap();
```

### Python

```python
import numpy as np
from koopman_dmd import DMD

# Oscillating signal, one row per variable
t = np.linspace(0, 10, 100)
data = np.vstack([np.sin(t), np.cos(t)])

# Fit DMD -- data is passed to the constructor, there is no separate fit() step
model = DMD(data, rank=2, dt=t[1] - t[0])

# Predict
predictions = model.predict(10)
print(f"Eigenvalues: {model.eigenvalues}")
print(f"Spectrum: {model.spectrum()}")
```

### R

```r
library(koopman.dmd)

# Oscillating signal
t <- seq(0, 10, length.out = 100)
X <- rbind(sin(t), cos(t))

# Fit DMD
result <- dmd(X, rank = 2)
summary(result)

# Predict
pred <- predict(result, n_ahead = 10)

# Spectrum and stability
spec <- dmd_spectrum(result, dt = 0.1)
stab <- dmd_stability(result)
```

## Advanced Usage

### DMD with Control (DMDc)

When the system is driven by a known input, standard DMD folds the forcing
into a biased `A`. `dmdc` identifies the forced linear system
`x_{t+1} = A x_t + B u_t` instead (Proctor, Brunton & Kutz 2016). Unlike
`dmd`, which takes one contiguous trajectory, `dmdc` takes explicit
snapshot-pair matrices -- `x1` (states at time `t`), `x2` (states one step
later), and `u` (the input applied during each transition) -- so columns may
come from many concatenated trajectories, and pairs may be freely masked out:

```rust
use koopman_dmd::{dmdc, DmdcConfig, stability_from_eigenvalues};

// Simulate a forced linear system: x_{t+1} = A0 x_t + B0 u_t
// with A0 = [[0.9, 0.1], [0.0, 0.8]], B0 = [0.5, 1.0].
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
println!("Spectral radius: {:.3}", stab.spectral_radius);
```

Two identification modes:

- **Unknown B** (default): jointly solves `[A B] = X2 Omega^+` with
  `Omega = [X1; U]`. Requires the input to be persistently exciting and
  *exogenous* -- fitting both `A` and `B` from closed-loop (state-feedback)
  data is biased and non-unique.
- **Known B** (`DmdcConfig { known_b: Some(b), .. }`): subtracts the known
  input response and solves only `A = (X2 - B U) X1^+`. Preferred whenever
  the input coupling is known by construction; immune to the closed-loop
  caveat.

With zero control rows (`u` of shape `0 x m`), `dmdc` performs autonomous
multi-trajectory identification from explicit pairs -- something `dmd` cannot
do. `DmdcConfig::rank_output` optionally projects the result onto the leading
SVD basis of `X2`, giving reduced operators `(A~, B~)` for model reduction.
The companions `stability_from_eigenvalues` and `spectrum_from_eigenvalues`
apply the standard analysis tools to the eigenvalues of the identified `A`.

The same functionality is exposed in the Python and R bindings:

```python
import koopman_dmd

d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)   # or known_b=B to pin B
d.a, d.b                    # identified matrices
d.stability()               # (is_stable, is_unstable, is_marginal, spectral_radius)
pred = d.predict(u=u_new)   # simulate under a new input sequence
```

```r
library(koopman.dmd)

fit <- dmdc(X1, X2, U, rank_input = 3)   # or known_B = B to pin B
fit$a; fit$b                             # identified matrices
dmdc_stability(fit)
pred <- predict(fit, U = U_new)          # simulate under a new input sequence
```

### Extended DMD with Lifting

Lifting maps observables into a higher-dimensional space where nonlinear dynamics become approximately linear:

```rust
use koopman_dmd::{dmd, DmdConfig, LiftingConfig};

let config = DmdConfig {
    lifting: Some(LiftingConfig::Polynomial { degree: 2 }),
    ..Default::default()
};
let result = dmd(&data, &config).unwrap();
```

### Hankel-DMD

Time-delay embedding for scalar signals or systems with limited measurements:

```rust
use koopman_dmd::{hankel_dmd, HankelConfig};

let config = HankelConfig {
    delays: Some(20),
    rank: Some(4),
    dt: 0.01,
};
let result = hankel_dmd(&signal, &config).unwrap();
```

### Generalized Laplace Analysis

Direct computation of Koopman eigenfunctions via weighted time averages:

```rust
use koopman_dmd::{gla, GlaConfig};

let config = GlaConfig {
    eigenvalues: None,    // auto-detect
    n_eigenvalues: 4,
    tol: 1e-6,
    max_iter: None,
};
let result = gla(&data, &config).unwrap();
```

### Harmonic Time Averages and Mesochronic Plots

Phase space analysis of area-preserving maps:

```rust
use koopman_dmd::*;

let map = StandardMap { epsilon: 0.12 };

// HTA at a single initial condition
let hta = harmonic_time_average(
    &[0.5, 0.3], &map, &Observable::SinPi, 0.5, 10000
).unwrap();

// Mesochronic plot over a grid (parallelized with rayon)
let mhp = mesochronic_compute(
    &map, (0.0, 1.0), (0.0, 1.0), 100,
    &Observable::SinPi, 0.5, 10000
).unwrap();
```

## Tests

```bash
# Rust (94 tests)
cargo test --workspace

# Python (52 tests)
cd koopman-dmd-py && python -m pytest tests/

# R (95 tests)
Rscript -e 'library(koopman.dmd); testthat::test_dir("koopman-dmd-r/tests/testthat")'
```

## Benchmarks

```bash
cargo bench -p koopman-dmd
```

Representative results (Apple Silicon):

| Operation | Size | Time |
|-----------|------|------|
| DMD | 5 x 100 | ~80 us |
| DMD | 50 x 1000 | ~8.5 ms |
| Predict (modes) | 2 vars, 100 steps | ~215 us |
| Predict (matrix) | 2 vars, 100 steps | ~36 us |
| Hankel-DMD | 1 x 200, 20 delays | ~130 us |
| GLA | 2 x 200 | ~344 us |

## References

- Schmid, P.J. (2010). Dynamic mode decomposition of numerical and experimental data. *Journal of Fluid Mechanics*, 656, 5-28.
- Kutz, J.N., Brunton, S.L., Brunton, B.W., & Proctor, J.L. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.
- Mezic, I. (2020). Spectrum of the Koopman operator, spectral expansions in functional spaces, and state-space geometry. arXiv:2009.05883.
- Levnajic, Z. & Mezic, I. (2014). Ergodic theory and visualization. arXiv:0808.2182v2.

## License

MIT
