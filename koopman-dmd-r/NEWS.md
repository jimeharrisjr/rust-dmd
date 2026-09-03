# koopman.dmd 0.2.2

* References in the `Description` field now use the CRAN-required
  `authors (year) <doi:...>` form, as requested in manual review. No code
  changes.

# koopman.dmd 0.2.1

* The Rust static archive is now removed from the build tree once the shared
  library is linked. R (>= 4.6.0) scans the symbol tables of linked static
  libraries, and the Rust standard library's bundled runtime objects reference
  `exit`/`abort` entry points that are never linked into the final shared
  library; leaving the archive behind produced a spurious `checking compiled
  code` warning on CRAN's Linux checks.

# koopman.dmd 0.2.0

* New `dmdc()`: Dynamic Mode Decomposition with control (Proctor, Brunton and
  Kutz, 2016), identifying the forced linear system `x_{t+1} = A x_t + B u_t`
  from snapshot pairs and control inputs. Supports joint estimation of `A` and
  `B`, a known-B mode that estimates only `A`, an optional reduced-order output
  projection, and autonomous multi-trajectory fits from explicit pairs
  (`U = NULL`).

* New `predict()` method for `dmdc` objects simulates the identified system
  under a given control input sequence, or its zero-input free response.

* The bundled Rust backend is updated to 'koopman-dmd' 0.2.0, which also fixes
  the full-A fallback and centered amplitudes in core DMD.

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
