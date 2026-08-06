# koopman.dmd 0.1.0

* Initial release.

* Core Dynamic Mode Decomposition (`dmd()`) with truncated SVD, optional mean
  centering, and automatic rank selection.

* Extended DMD via lifting functions (`polynomial`, `polynomial_cross`,
  `trigonometric`, `delay`), for systems whose dynamics are nonlinear in the
  original coordinates.

* Hankel-DMD (`hankel_dmd()`) using time-delay embedding, for scalar or
  low-dimensional signals.

* Generalized Laplace Analysis (`gla()`) for computing Koopman eigenfunctions
  directly from weighted time averages.

* Harmonic time averages (`harmonic_time_average()`, `hta_convergence()`) and
  mesochronic harmonic plots (`mesochronic_compute()`) for phase space analysis,
  with orbit classification via `classify_phase_space()`.

* Built-in maps for experimentation: Chirikov standard, Froeschlé, extended
  standard, Hénon, and logistic.

* Analysis helpers: `dmd_spectrum()`, `dmd_stability()`, `dmd_error()`,
  `dmd_residual()`, `dmd_dominant_modes()`, and `dmd_reconstruct()`.

* Numerics are implemented in Rust and reached through 'extendr'. The complete
  Rust dependency tree is bundled in `src/rust/vendor.tar.xz`, so the package
  builds without network access.
