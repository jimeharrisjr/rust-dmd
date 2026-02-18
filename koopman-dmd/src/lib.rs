//! # koopman-dmd
//!
//! Dynamic Mode Decomposition with Koopman operator theory extensions.
//!
//! This library provides data-driven analysis of dynamical systems using
//! DMD and related spectral methods:
//!
//! - **Core DMD** ([`dmd()`]): Standard DMD with truncated SVD, optional centering
//! - **Extended DMD** ([`lift_data`], [`LiftingConfig`]): Polynomial, trigonometric,
//!   and delay-coordinate lifting for nonlinear systems
//! - **Hankel-DMD** ([`hankel_dmd`]): Time-delay embedding via Krylov subspace
//! - **GLA** ([`gla()`]): Generalized Laplace Analysis for direct eigenfunction computation
//! - **Harmonic Time Averages** ([`harmonic_time_average`]): Phase space analysis via HTA
//! - **Mesochronic Plots** ([`mesochronic_compute`]): Parallelized grid HTA visualization
//! - **Built-in maps** ([`StandardMap`], [`HenonMap`], etc.): Chirikov, Froeschlé, Hénon, logistic
//!
//! ## Quick Start
//!
//! ```rust
//! use koopman_dmd::{dmd, DmdConfig, predict_modes};
//!
//! // Create a simple oscillating signal
//! let n = 100;
//! let mut data = faer::Mat::<f64>::zeros(2, n);
//! for j in 0..n {
//!     let t = j as f64 * 0.1;
//!     data[(0, j)] = t.sin();
//!     data[(1, j)] = t.cos();
//! }
//!
//! // Compute DMD
//! let config = DmdConfig::default();
//! let result = dmd(&data, &config).unwrap();
//!
//! // Predict 10 steps ahead
//! let pred = predict_modes(&result, 10, None).unwrap();
//! ```
//!
//! ## References
//!
//! - Schmid (2010), *J. Fluid Mech.*, 656, 5-28
//! - Kutz et al. (2016), *Dynamic Mode Decomposition*, SIAM
//! - Mezic (2020), arXiv:2009.05883
//! - Levnajic & Mezic (2014), arXiv:0808.2182v2

pub mod lifting;
pub mod types;

pub mod analysis;
pub mod dmd;
pub mod gla;
pub mod hankel;
pub mod harmonic;
pub mod maps;
pub mod mesochronic;
pub mod predict;
pub mod utils;

pub use analysis::{
    dmd_convergence, dmd_dominant_modes, dmd_error, dmd_pseudospectrum, dmd_reconstruct,
    dmd_residual, dmd_spectrum, dmd_stability,
};
pub use dmd::dmd;
pub use gla::{gla, gla_predict, gla_reconstruct, GlaConfig, GlaResult};
pub use hankel::{
    build_hankel_matrix, hankel_dmd, hankel_predict, hankel_reconstruct, HankelConfig,
    HankelDmdResult,
};
pub use harmonic::{
    classify_phase_space, harmonic_time_average, hta_convergence, hta_from_values, DynamicsType,
    HtaConvergenceResult, HtaResult, Observable, PhaseSpaceClass,
};
pub use lifting::{lift_data, LiftingConfig, LiftingInfo};
pub use maps::{
    generate_phase_grid, generate_trajectory, ClosureMap, ExtendedStandardMap, FroeschleMap,
    HenonMap, LogisticMap, MapFn, StandardMap,
};
pub use mesochronic::{
    mesochronic_compute, mesochronic_scatter, mesochronic_section, MhpResult, MhspResult,
};
pub use predict::{predict_matrix, predict_modes};
pub use types::{
    ConvergenceResult, DmdConfig, DmdError, DmdResult, DominantCriterion, ErrorMetrics, ModeInfo,
    PseudospectrumResult, ResidualResult, Stability, StabilityResult, C64,
};
