# Rust Koopman DMD - Phased Implementation Plan

## Overview

Port the RKoopmanDMD R package to a Rust library (`koopman-dmd`) with R and Python bindings. The R package (v0.6.0) implements Dynamic Mode Decomposition with Koopman operator theory extensions, including lifting functions (Extended DMD), Hankel-DMD, Generalized Laplace Analysis, harmonic time averages, and mesochronic plots.

**Architecture**: Rust core library + `koopman-dmd-r` (R bindings via extendr) + `koopman-dmd-py` (Python bindings via PyO3).

---

## Project Structure

```
rust-dmd/
├── Cargo.toml                    # Workspace root
├── koopman-dmd/                  # Core Rust library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── dmd.rs                # Core DMD algorithm
│       ├── lifting.rs            # Lifting functions (EDMD)
│       ├── predict.rs            # Prediction methods
│       ├── analysis.rs           # Spectrum, stability, error, residuals
│       ├── hankel.rs             # Hankel-DMD (Krylov subspace)
│       ├── gla.rs                # Generalized Laplace Analysis
│       ├── harmonic.rs           # Harmonic time averages
│       ├── mesochronic.rs        # Mesochronic harmonic plots
│       ├── maps.rs               # Built-in dynamical systems
│       ├── types.rs              # Core data structures
│       └── utils.rs              # Utilities (pinv, safe_solve, etc.)
├── koopman-dmd-py/               # Python bindings (PyO3)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs
├── koopman-dmd-r/                # R bindings (extendr)
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
├── examples/                     # Rust examples
├── tests/                        # Integration tests
├── benches/                      # Benchmarks
└── data/                         # Sample data (satellite)
```

---

## Dependency Stack

### Core Rust Dependencies
| Crate | Purpose |
|-------|---------|
| `ndarray` | N-dimensional arrays (matrix operations) |
| `ndarray-linalg` | SVD, eigendecomposition, solve (wraps LAPACK) |
| `num-complex` | Complex number arithmetic |
| `num-traits` | Numeric trait abstractions |
| `thiserror` | Error type definitions |
| `serde` | Serialization (optional, for data I/O) |
| `rayon` | Parallel computation (MHP grid, etc.) |

### Binding Dependencies
| Crate | Purpose |
|-------|---------|
| `extendr-api` | R bindings |
| `pyo3` | Python bindings |
| `numpy` (PyO3) | NumPy array interop |

---

## Phase 1: Foundation & Core DMD

**Goal**: Working Rust library with basic DMD, prediction, and spectrum analysis.

### 1.1 Project Scaffolding
- [ ] Initialize Cargo workspace with `koopman-dmd` library crate
- [ ] Set up `ndarray`, `ndarray-linalg`, `num-complex` dependencies
- [ ] Configure CI (clippy, rustfmt, tests)
- [ ] Define error types (`DmdError` enum via `thiserror`)

### 1.2 Core Types (`types.rs`)
- [ ] `DmdResult` struct — mirrors R's `dmd` object:
  - `a_matrix: Array2<Complex64>` — full Koopman operator approximation
  - `modes: Array2<Complex64>` — DMD modes (Φ)
  - `eigenvalues: Array1<Complex64>` — eigenvalues (λ)
  - `amplitudes: Array1<Complex64>` — initial amplitudes (b)
  - `rank: usize`
  - `svd: SvdComponents` — U, S, V from truncated SVD
  - `a_tilde: Array2<Complex64>` — reduced DMD matrix
  - `x_first: Array1<f64>`, `x_last: Array1<f64>`
  - `data_dim: (usize, usize)` — (n_vars, n_time)
  - `center: bool`, `x_mean: Option<Array1<f64>>`
  - `dt: f64`
  - Lifting metadata (phase 2)

### 1.3 Utilities (`utils.rs`)
- [ ] `validate_matrix()` — input validation (min rows/cols, NaN/Inf check)
- [ ] `determine_rank()` — auto rank selection (99% variance threshold from singular values)
- [ ] `pinv()` — Moore-Penrose pseudo-inverse via SVD
- [ ] `safe_solve()` — linear solve with pseudo-inverse fallback

### 1.4 Core DMD Algorithm (`dmd.rs`)
- [ ] `dmd(x: &Array2<f64>, config: DmdConfig) -> Result<DmdResult, DmdError>`
  - `DmdConfig` struct: `rank: Option<usize>`, `center: bool`, `dt: f64`
  - Split data: X₁ = X[:, 0..n-1], X₂ = X[:, 1..n]
  - Optional centering (subtract row means)
  - Truncated SVD of X₁
  - Reduced operator: Ã = Uᵀ X₂ V Σ⁻¹
  - Eigendecomposition of Ã
  - Physical modes: Φ = X₂ V Σ⁻¹ W
  - Amplitudes: b = Φ⁺ x₀
  - Full operator: A = Φ Λ Φ⁻¹
- [ ] Unit tests: known oscillatory system, rank selection, centering

### 1.5 Prediction (`predict.rs`)
- [ ] `predict_modes(result: &DmdResult, n_ahead: usize, x0: Option<&Array1<f64>>) -> Array2<f64>`
  - x(k) = Σᵢ φᵢ · bᵢ · λᵢᵏ
- [ ] `predict_matrix(result: &DmdResult, n_ahead: usize, x0: Option<&Array1<f64>>) -> Array2<f64>`
  - Iterative: x(k+1) = A · x(k)
- [ ] Tests: compare both methods, verify reconstruction of training data

### 1.6 Spectrum Analysis (`analysis.rs`)
- [ ] `dmd_spectrum(result: &DmdResult, dt: f64) -> Vec<ModeInfo>`
  - `ModeInfo`: magnitude, phase, frequency, period, growth_rate, half_life, stability
- [ ] `dmd_stability(result: &DmdResult, tol: f64) -> StabilityResult`
  - Per-mode classification: decaying / neutral / growing
  - Overall: is_stable, spectral_radius
- [ ] `dmd_reconstruct(result: &DmdResult, n_steps: usize, modes: Option<&[usize]>) -> Array2<f64>`
- [ ] `dmd_error(result: &DmdResult, x_original: &Array2<f64>) -> ErrorMetrics`
  - RMSE, MAE, MAPE, relative_error, per-variable RMSE
- [ ] `dmd_dominant_modes(result: &DmdResult, n: usize, criterion: Criterion) -> Vec<usize>`
  - Criteria: amplitude, energy (amplitude × magnitude), stability
- [ ] Tests against R package outputs on known systems

### Phase 1 Deliverable
A Rust library that can: decompose a time-series matrix, extract modes/eigenvalues, predict future states, and analyze spectral properties. Verified against R package results.

---

## Phase 2: Lifting Functions (Extended DMD)

**Goal**: Support nonlinear dynamics via observable lifting.

### 2.1 Lifting Infrastructure (`lifting.rs`)
- [ ] `LiftingFn` trait:
  ```rust
  trait LiftingFn {
      fn lift(&self, x: &Array2<f64>) -> Array2<f64>;
      fn name(&self) -> &str;
      fn output_dim(&self, input_dim: usize) -> usize;
  }
  ```
- [ ] `LiftingConfig` enum for specifying lifting via config:
  - `Polynomial { degree: usize }`
  - `PolynomialCross { degree: usize }`
  - `Trigonometric { harmonics: usize }`
  - `Delay { delays: usize }`
  - `Rbf { centers: Array2<f64>, sigma: f64 }`
  - `Custom(Box<dyn LiftingFn>)`

### 2.2 Built-in Lifting Functions
- [ ] `lift_polynomial(x, degree)` — [X, X², X³, ...]
- [ ] `lift_poly_cross(x, degree)` — polynomial with all cross-terms
- [ ] `lift_trigonometric(x, harmonics)` — [X, sin(X), cos(X), sin(2X), ...]
- [ ] `lift_delay(x, delays)` — time-delay coordinates
- [ ] `lift_rbf(x, centers, sigma)` — Gaussian RBF features

### 2.3 Integration with DMD
- [ ] Extend `DmdConfig` with `lifting: Option<LiftingConfig>`, `observables: Option<Vec<usize>>`
- [ ] Extend `DmdResult` with lifting metadata: `lifting_fn`, `n_vars_original`, `n_vars_lifted`
- [ ] Prediction in original space (project back from lifted)
- [ ] Tests: nonlinear system (e.g., x² dynamics) with polynomial lifting vs. without

### Phase 2 Deliverable
Extended DMD that handles nonlinear systems by lifting to higher-dimensional observable spaces.

---

## Phase 3: Hankel-DMD & GLA

**Goal**: Time-delay embedding DMD and direct eigenfunction computation.

### 3.1 Hankel-DMD (`hankel.rs`)
- [ ] `HankelDmdResult` struct (extends DmdResult):
  - `hankel: Array2<f64>` — Hankel-Takens matrix
  - `delays: usize`
  - `companion: Array2<f64>`
  - `residual: f64`
- [ ] `hankel_dmd(y: &Array1<f64>, config: HankelConfig) -> Result<HankelDmdResult, DmdError>`
  - Construct Hankel matrix from delayed copies
  - Apply standard DMD to Hankel matrix
  - Extract companion matrix
- [ ] `hankel_reconstruct()` — reconstruction from Hankel modes
- [ ] Prediction methods adapted for Hankel structure
- [ ] Tests: scalar oscillatory signal, verify frequency recovery

### 3.2 Generalized Laplace Analysis (`gla.rs`)
- [ ] `GlaResult` struct:
  - `eigenvalues: Array1<Complex64>`
  - `eigenfunctions: Array2<Complex64>` — (n_eig × n_time)
  - `modes: Array2<Complex64>` — (n_obs × n_eig)
  - `convergence: Vec<bool>`, `residuals: Vec<f64>`
- [ ] `gla(y: &Array2<f64>, config: GlaConfig) -> Result<GlaResult, DmdError>`
  - Direct eigenfunction computation via weighted time averages
  - Orthogonal subtraction of previously found eigenfunctions
  - Convergence checking
- [ ] `gla_reconstruct()`, `predict_gla()`
- [ ] Tests: system with known eigenvalues, convergence verification

### 3.3 Advanced Analysis Extensions
- [ ] `dmd_residual()` — residual analysis (Mezic 2020 error bounds)
- [ ] `dmd_pseudospectrum()` — ε-pseudospectrum (min σ of (zI - A) over grid)
- [ ] `dmd_convergence()` — convergence rate estimation vs. sample size

### Phase 3 Deliverable
Complete suite of Koopman approximation methods: standard DMD, EDMD (lifting), Hankel-DMD, and GLA.

---

## Phase 4: Harmonic Analysis & Mesochronic Plots

**Goal**: Phase space visualization tools for Hamiltonian and dissipative systems.

### 4.1 Built-in Dynamical Systems (`maps.rs`)
- [ ] `standard_map(state, epsilon)` — Chirikov standard map
- [ ] `froeschle_map(state, epsilon1, epsilon2, eta)` — 4D coupled
- [ ] `extended_standard_map(state, epsilon, delta)` — 3D
- [ ] `henon_map(state, a, b)` — 2D dissipative
- [ ] `logistic_map(x, r)` — 1D
- [ ] `generate_trajectory(x0, map_fn, n_iter)` — orbit generation
- [ ] `MapFn` trait for user-defined maps

### 4.2 Harmonic Time Averages (`harmonic.rs`)
- [ ] `harmonic_time_average(trajectory, observable, omega/period) -> HtaResult`
  - f*_ω(x) = (1/T) Σ exp(i·2πkω) · f(Tᵏx)
  - Returns: complex value, magnitude, phase
- [ ] Built-in observables: identity, sin_pi, cos_pi, sin_2pi, cos_2pi
- [ ] `hta_convergence()` — convergence analysis (periodic vs. chaotic detection)
- [ ] `Observable` trait for custom observables

### 4.3 Mesochronic Harmonic Plots (`mesochronic.rs`)
- [ ] `MhpResult` struct:
  - `hta_matrix: Array2<f64>` — |HTA| magnitudes over grid
  - `phase_matrix: Array2<f64>` — arg(HTA) phases
  - `x_coords, y_coords: Array1<f64>`
  - Metadata: omega, period, n_iter, map_params
- [ ] `mesochronic_plot(map_fn, x_range, y_range, resolution, config) -> MhpResult`
  - Parallel grid computation via `rayon`
  - Per-point: generate trajectory, compute HTA
- [ ] `mesochronic_scatter_plot()` — multi-observable variant
- [ ] `mesochronic_section_plot()` — 2D sections of higher-D systems
- [ ] Phase space classification: periodic islands, KAM tori, chaotic seas

### Phase 4 Deliverable
Complete phase space analysis toolkit with parallelized computation, matching R package's mesochronic visualization capabilities.

---

## Phase 5: Python Bindings (PyO3)

**Goal**: Full Python API via `koopman-dmd-py` package.

### 5.1 Setup
- [ ] PyO3 + maturin project structure
- [ ] NumPy interop via `numpy` crate (zero-copy where possible)
- [ ] `pyproject.toml` with build configuration

### 5.2 Python API Surface
- [ ] `DMD` class wrapping `DmdResult`
  - `DMD(X, rank=None, center=False, lifting=None, dt=1.0)`
  - Properties: `modes`, `eigenvalues`, `amplitudes`, `rank`, `A`
  - Methods: `predict(n_ahead, x0=None, method="modes")`, `reconstruct()`, `spectrum(dt=None)`
  - `__repr__`, `__str__`
- [ ] `HankelDMD` class wrapping `HankelDmdResult`
- [ ] `GLA` class wrapping `GlaResult`
- [ ] Analysis functions: `dmd_stability()`, `dmd_error()`, `dmd_dominant_modes()`, etc.
- [ ] Lifting: `PolynomialLifting`, `TrigLifting`, etc. as Python classes
- [ ] Maps: `standard_map()`, `henon_map()`, `generate_trajectory()`
- [ ] HTA/MHP: `harmonic_time_average()`, `mesochronic_plot()`
- [ ] Exception hierarchy: `DmdError`, `DmdValueError`, `DmdConvergenceError`

### 5.3 Testing & Documentation
- [ ] pytest suite mirroring R test cases
- [ ] Type stubs (`.pyi` files) for IDE support
- [ ] Docstrings with NumPy-style formatting
- [ ] Example notebooks (Jupyter)

### Phase 5 Deliverable
`pip install koopman-dmd` with Pythonic API, NumPy interop, and comprehensive documentation.

---

## Phase 6: R Bindings (extendr)

**Goal**: Drop-in R package powered by Rust backend.

### 6.1 Setup
- [ ] extendr project structure within `koopman-dmd-r/`
- [ ] R package wrapper (`koopman-dmd-r/R/`) with roxygen2 docs
- [ ] DESCRIPTION, NAMESPACE configuration

### 6.2 R API Surface
Goal: API-compatible with RKoopmanDMD where possible.
- [ ] `dmd()` — calls Rust, returns S3 "dmd" object
- [ ] `predict.dmd()`, `print.dmd()`, `summary.dmd()`, `plot.dmd()`
- [ ] `hankel_dmd()`, `gla()`
- [ ] Analysis functions: `dmd_spectrum()`, `dmd_stability()`, `dmd_error()`, etc.
- [ ] Lifting functions: same interface as RKoopmanDMD
- [ ] Maps and HTA/MHP functions
- [ ] Backward-compatible with existing RKoopmanDMD user code

### 6.3 Testing & Validation
- [ ] testthat suite comparing Rust-backed results to pure R results
- [ ] Numerical tolerance checks (R vs Rust may differ in floating-point edge cases)
- [ ] Performance benchmarks vs. pure R

### Phase 6 Deliverable
R package installable via `devtools::install()` that accelerates all RKoopmanDMD computations via Rust.

---

## Phase 7: Polish & Release

### 7.1 Documentation
- [ ] Rust API docs (`cargo doc`)
- [ ] User guide with mathematical background
- [ ] Migration guide from RKoopmanDMD

### 7.2 Performance
- [ ] Benchmarks: Rust vs R on satellite data, large systems
- [ ] SIMD optimizations where applicable (via BLAS backend)
- [ ] Memory profiling for large-scale DMD

### 7.3 Testing
- [ ] Cross-validation: R ↔ Rust ↔ Python results match
- [ ] Edge cases: rank-1 systems, single-variable, very large matrices
- [ ] Numerical stability tests (near-singular, ill-conditioned)

### 7.4 Packaging
- [ ] Publish to crates.io (`koopman-dmd`)
- [ ] Publish to PyPI (`koopman-dmd`)
- [ ] Publish to CRAN or R-universe
- [ ] CI/CD for all three ecosystems

---

## Key Mathematical References

1. **Schmid (2010)** — DMD of numerical and experimental data. *J. Fluid Mech.*, 656, 5-28.
2. **Kutz et al. (2016)** — *Dynamic Mode Decomposition*. SIAM.
3. **Williams, Kevrekidis, Rowley (2015)** — Data-driven Koopman operator approximation. *J. Nonlinear Sci.*, 25(6), 1307-1346.
4. **Mezic (2020)** — On numerical approximations of the Koopman operator. arXiv:2009.05883.
5. **Levnajic & Mezic (2014)** — Harmonic mesochronic plots. arXiv:0808.2182v2.

---

## Risk Factors & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `ndarray-linalg` LAPACK linking issues | Blocks all phases | Use `openblas-static` feature; fallback to `faer` crate |
| Numerical divergence from R results | Blocks validation | Use identical algorithms; tolerance-based comparison |
| extendr API instability | Blocks R bindings | Pin versions; consider savvy as alternative |
| Complex eigenvalue handling | Correctness issues | Use `num-complex` consistently; test conjugate pairs |
| Large matrix memory | Performance | Use views/slices; consider sparse representations |

---

## Priority Order

Phases 1-3 are the mathematical core and should be implemented sequentially. Phases 5 and 6 (Python and R bindings) can be developed in parallel once Phase 3 is complete. Phase 4 (harmonic/mesochronic) is independent of bindings and can be developed concurrently with Phase 5/6.

```
Phase 1 (Core DMD) → Phase 2 (Lifting) → Phase 3 (Hankel + GLA)
                                                ↓
                                    ┌───────────┼───────────┐
                                    ↓           ↓           ↓
                              Phase 4       Phase 5     Phase 6
                              (HTA/MHP)    (Python)      (R)
                                    └───────────┼───────────┘
                                                ↓
                                          Phase 7 (Polish)
```
