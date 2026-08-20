# koopman-dmd

[![PyPI](https://img.shields.io/pypi/v/koopman-dmd.svg)](https://pypi.org/project/koopman-dmd/)
[![Python versions](https://img.shields.io/pypi/pyversions/koopman-dmd.svg)](https://pypi.org/project/koopman-dmd/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Dynamic Mode Decomposition (DMD) with Koopman operator theory extensions, with a Rust core.

DMD extracts spatiotemporal coherent structures from time-series data, giving a linear
operator that approximates the dynamics of a possibly nonlinear system. This package wraps
the [`koopman-dmd`](https://crates.io/crates/koopman-dmd) Rust crate via PyO3, so the
numerics run at native speed with no BLAS/LAPACK installation required.

## Installation

```bash
pip install koopman-dmd
```

The distribution is named `koopman-dmd`; the import name is `koopman_dmd`:

```python
import koopman_dmd
```

Prebuilt wheels are published for Linux (x86_64, aarch64), macOS (x86_64, arm64), and
Windows (x86_64) on Python 3.9+. Installing from source requires a Rust toolchain.

## Quick start

Data is passed as a NumPy array with **one row per variable and one column per time step**.

```python
import numpy as np
import koopman_dmd

# A 2-variable oscillating signal, shape (2, 100)
t = np.linspace(0, 10, 100)
x = np.vstack([np.sin(t), np.cos(t)])

d = koopman_dmd.DMD(x, rank=2, dt=t[1] - t[0])

print(d.eigenvalues)      # (r, 2) array of [real, imag]
print(d.modes)            # DMD modes
print(d.singular_values)

# Forecast 10 steps beyond the input
future = d.predict(10)

# Per-mode frequency, growth rate, amplitude, and stability
for mode in d.spectrum():
    print(mode)
```

Note that the data is passed to the constructor — there is no separate `fit()` step.

## `DMD`

```python
koopman_dmd.DMD(x, rank=None, center=False, dt=1.0, lifting=None, lifting_param=None)
```

`rank=None` selects a truncation rank automatically (99% of variance). `lifting` enables
Extended DMD and accepts `"polynomial"`, `"polynomial_cross"`, `"trigonometric"`, or
`"delay"`, with the degree / harmonic count / delay count given by `lifting_param`.

**Properties:** `rank`, `data_dim`, `center`, `dt`, `eigenvalues`, `modes`, `amplitudes`,
`singular_values`

**Methods:**

| Method | Returns |
|---|---|
| `predict(n_ahead, x0=None, method="modes")` | forecast array; `method` is `"modes"` or `"matrix"` |
| `reconstruct(n_steps, modes_subset=None)` | reconstruction from all or selected modes |
| `spectrum()` | per-mode frequency, growth rate, amplitude, stability — uses the `dt` given at construction |
| `stability()` | `(is_stable, is_unstable, is_marginal, spectral_radius)` |
| `error()` | `(rmse, mae, mape, rel_err)` |
| `dominant_modes(n, criterion="amplitude")` | indices of the `n` most significant modes |
| `residual()` | `(absolute, relative)` |

### Extended DMD with lifting

```python
d = koopman_dmd.DMD(x, lifting="polynomial", lifting_param=2)
```

## `DMDc`

DMD with control (Proctor, Brunton & Kutz 2016) — identifies the forced linear system
`x_{t+1} = A x_t + B u_t` from snapshot pairs and control inputs.

```python
koopman_dmd.DMDc(x1, x2, u=None, rank_input=None, rank_output=None, dt=1.0, known_b=None)
```

Unlike `DMD`, which takes one contiguous trajectory, `DMDc` takes explicit pair
matrices: `x1` holds states at time `t`, `x2` the states one step later, and `u` the
input applied during each transition, so columns may come from many concatenated
trajectories. `u=None` fits an autonomous multi-trajectory model from the pairs.
Passing `known_b` pins the input matrix and estimates only `A` — preferred whenever the
input coupling is known by construction, and required for closed-loop (state-feedback)
data, where joint identification is biased. `rank_output` optionally projects onto the
leading SVD basis of `x2`, giving reduced operators `a_tilde`, `b_tilde` for model
reduction.

**Properties:** `a`, `b`, `a_tilde`, `b_tilde`, `basis`, `eigenvalues`,
`singular_values`, `rank_input`, `rank_output`, `dt`, `n_states`, `n_inputs`

**Methods:**

| Method | Returns |
|---|---|
| `predict(u=None, x0=None, n_ahead=None)` | states from stepping `x_{t+1} = A x_t + B u_t`; the columns of `u` set the horizon, or `n_ahead` steps of zero input |
| `spectrum()` | per-mode frequency, growth rate, stability of the identified operator |
| `stability()` | `(is_stable, is_unstable, is_marginal, spectral_radius)` |

```python
# Recover A and B from a forced linear system driven by a probe input
d = koopman_dmd.DMDc(x1, x2, u, rank_input=3)
print(d.a)                     # state-transition matrix
print(d.b)                     # input matrix
pred = d.predict(u=u_future)   # simulate under a new input sequence
```

The input must be persistently exciting (and not pure state feedback) for the joint
identification to be well-posed.

## `HankelDMD`

Time-delay embedding, for scalar signals or systems with few measured variables.

```python
koopman_dmd.HankelDMD(y, delays=None, rank=None, dt=1.0)
```

**Properties:** `rank`, `delays`, `n_obs`, `residual`, `eigenvalues`
**Methods:** `predict(n_ahead)`, `reconstruct(n_steps)`

```python
y = np.sin(np.linspace(0, 4 * np.pi, 200)).reshape(1, -1)
h = koopman_dmd.HankelDMD(y, delays=20)
print(h.rank)          # 2 — chosen automatically
print(h.predict(10))
```

Leaving `rank=None` is usually right. Requesting a rank higher than the signal actually
supports (a pure sinusoid is rank 2) leaves the reduced operator near-singular, and the
eigendecomposition can fail with `NoConvergence`.

## `GLA`

Generalized Laplace Analysis — computes Koopman eigenfunctions directly via weighted
time averages.

```python
koopman_dmd.GLA(y, eigenvalues=None, n_eigenvalues=5, tol=1e-6, max_iter=None)
```

**Properties:** `n_obs`, `n_time`, `eigenvalues`, `convergence`, `residuals`
**Methods:** `predict(n_ahead)`, `reconstruct(modes_to_use=None)`

## Phase space analysis

For area-preserving and chaotic maps.

**Maps** and their `params` keys — `params` is a dict, and omitting it uses defaults:

| `map_name` | State dim | Parameters |
|---|---|---|
| `"standard"` (Chirikov) | 2 | `epsilon` |
| `"froeschle"` | 4 | `epsilon`, `eta` |
| `"extended_standard"` | 3 | `epsilon`, `delta` |
| `"henon"` | 2 | `a`, `b` |
| `"logistic"` | 1 | `r` |

**Observables:** `"identity"`, `"sin_pi"`, `"cos_pi"`, `"sin_pi_xy"`, `"cos_pi_xy"`,
`"sin_2pi"`, `"cos_2pi"`, `"trig_product"`

Initial conditions are NumPy arrays, not lists.

```python
ic = np.array([0.5, 0.3])

# Iterate a map from an initial condition
traj = koopman_dmd.generate_trajectory("standard", ic, 1000, {"epsilon": 0.12})

# Harmonic time average at a single initial condition
mag, phase, hta_re, hta_im = koopman_dmd.harmonic_time_average(
    "standard", ic, "sin_pi", 0.5, 10000, {"epsilon": 0.12}
)

# Mesochronic plot over a grid (parallelized in Rust).
# Returns (hta_magnitude, phase, x_coords, y_coords).
hta, phase, x_coords, y_coords = koopman_dmd.mesochronic_compute(
    "standard", (0.0, 1.0), (0.0, 1.0), 100, "sin_pi", 0.5, 10000, {"epsilon": 0.12}
)

# Classify orbits as regular, resonating, or chaotic from HTA magnitudes
labels = koopman_dmd.classify_phase_space(hta.ravel())

# Convergence history of the time average
conv = koopman_dmd.hta_convergence("standard", ic, "sin_pi", 0.5, 10000, {"epsilon": 0.12})
```

## Other languages

- **Rust** — [`koopman-dmd` on crates.io](https://crates.io/crates/koopman-dmd)
- **R** — `koopman.dmd` (extendr)

## Development

```bash
pip install maturin pytest numpy
maturin develop --release
pytest tests/
```

## References

- Schmid, P.J. (2010). Dynamic mode decomposition of numerical and experimental data.
  *Journal of Fluid Mechanics*, 656, 5–28.
- Kutz, J.N., Brunton, S.L., Brunton, B.W., & Proctor, J.L. (2016).
  *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.
- Mezić, I. (2020). Spectrum of the Koopman operator, spectral expansions in functional
  spaces, and state-space geometry. [arXiv:2009.05883](https://arxiv.org/abs/2009.05883)
- Levnajić, Z. & Mezić, I. (2014). Ergodic theory and visualization.
  [arXiv:0808.2182v2](https://arxiv.org/abs/0808.2182)

## License

MIT — see [LICENSE](LICENSE).
