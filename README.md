# koopman-dmd

Dynamic Mode Decomposition with Koopman operator theory extensions, implemented in Rust with Python and R bindings.

## Features

- **Core DMD** -- Standard Dynamic Mode Decomposition with truncated SVD and optional mean centering
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
koopman-dmd = { git = "https://github.com/jimeharrisjr/rust-dmd" }
```

### Python

Requires Rust toolchain and maturin:

```bash
pip install maturin
cd koopman-dmd-py
maturin develop --release
```

### R

Requires Rust toolchain:

```r
install.packages("koopman-dmd-r", repos = NULL, type = "source")
```

Or from the repository:

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

# Oscillating signal
t = np.linspace(0, 10, 100)
data = np.vstack([np.sin(t), np.cos(t)])

# Fit DMD
model = DMD(rank=2)
model.fit(data)

# Predict
predictions = model.predict(10)
print(f"Eigenvalues: {model.eigenvalues}")
print(f"Spectrum: {model.spectrum(dt=0.1)}")
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
