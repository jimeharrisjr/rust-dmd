use faer::Mat;

use crate::lifting::{LiftingConfig, LiftingInfo};

/// Error types for DMD operations.
#[derive(Debug, thiserror::Error)]
pub enum DmdError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("SVD computation failed: {0}")]
    SvdFailed(String),

    #[error("eigendecomposition failed: {0}")]
    EigenFailed(String),

    #[error("linear solve failed: {0}")]
    SolveFailed(String),

    #[error("numerical error: {0}")]
    NumericalError(String),
}

/// Configuration for DMD computation.
#[derive(Debug, Clone)]
pub struct DmdConfig {
    /// Truncation rank. None for automatic selection (99% variance).
    pub rank: Option<usize>,
    /// Whether to center the data (subtract row means).
    pub center: bool,
    /// Time step between snapshots.
    pub dt: f64,
    /// Optional lifting transformation for Extended DMD.
    pub lifting: Option<LiftingConfig>,
}

impl Default for DmdConfig {
    fn default() -> Self {
        Self {
            rank: None,
            center: false,
            dt: 1.0,
            lifting: None,
        }
    }
}

/// Components of the truncated SVD.
#[derive(Debug, Clone)]
pub struct SvdComponents {
    /// Left singular vectors (m × r).
    pub u: Mat<f64>,
    /// Singular values (r), stored as a column vector.
    pub s: Vec<f64>,
    /// Right singular vectors (n × r), columns are right singular vectors.
    pub v: Mat<f64>,
}

/// Complex number type (re, im).
#[derive(Debug, Clone, Copy)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    /// Create a new complex number.
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Magnitude |z| = sqrt(re² + im²).
    pub fn norm(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Squared magnitude re² + im².
    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle atan2(im, re).
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate (re, -im).
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Raise to a real power: (r e^{iθ})^p = r^p e^{ipθ}.
    pub fn powf(&self, p: f64) -> Self {
        let r = self.norm();
        let theta = self.arg();
        let rp = r.powf(p);
        Self {
            re: rp * (p * theta).cos(),
            im: rp * (p * theta).sin(),
        }
    }

    /// The zero complex number (0 + 0i).
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
}

impl std::ops::Add for C64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::AddAssign for C64 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::Sub for C64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for C64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for C64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::Div for C64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl std::ops::Div<f64> for C64 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

/// Result of a DMD computation.
#[derive(Debug, Clone)]
pub struct DmdResult {
    /// Full Koopman operator approximation (m × m), complex.
    pub a_matrix: Vec<Vec<C64>>,
    /// DMD modes Φ (m × r), columns are modes.
    pub modes: Vec<Vec<C64>>,
    /// Eigenvalues λ (r).
    pub eigenvalues: Vec<C64>,
    /// Initial amplitudes b (r).
    pub amplitudes: Vec<C64>,
    /// Truncation rank used.
    pub rank: usize,
    /// Truncated SVD components.
    pub svd: SvdComponents,
    /// Reduced DMD matrix Ã (r × r), complex.
    pub a_tilde: Vec<Vec<C64>>,
    /// First snapshot (m).
    pub x_first: Vec<f64>,
    /// Last snapshot (m).
    pub x_last: Vec<f64>,
    /// Data dimensions (n_vars, n_time).
    pub data_dim: (usize, usize),
    /// Whether data was centered.
    pub center: bool,
    /// Row means (if centered).
    pub x_mean: Option<Vec<f64>>,
    /// Time step.
    pub dt: f64,
    /// Lifting metadata (if lifting was applied).
    pub lifting_info: Option<LiftingInfo>,
}

impl DmdResult {
    /// Number of original (pre-lifting) state variables.
    pub fn n_vars_original(&self) -> usize {
        match &self.lifting_info {
            Some(info) => info.n_vars_original,
            None => self.data_dim.0,
        }
    }

    /// Whether lifting was applied.
    pub fn is_lifted(&self) -> bool {
        self.lifting_info.is_some()
    }
    /// Get mode column j as a slice of C64.
    pub fn mode(&self, j: usize) -> Vec<C64> {
        let n_vars = self.data_dim.0;
        (0..n_vars).map(|i| self.modes[i][j]).collect()
    }

    /// Number of state variables.
    pub fn n_vars(&self) -> usize {
        self.data_dim.0
    }
}

/// Information about a single DMD mode.
#[derive(Debug, Clone)]
pub struct ModeInfo {
    /// Mode index.
    pub index: usize,
    /// Complex eigenvalue.
    pub eigenvalue: C64,
    /// Eigenvalue magnitude |λ|.
    pub magnitude: f64,
    /// Eigenvalue phase angle (radians).
    pub phase: f64,
    /// Oscillation frequency (cycles per dt).
    pub frequency: f64,
    /// Oscillation period (in dt units).
    pub period: f64,
    /// Growth rate (log|λ|/dt).
    pub growth_rate: f64,
    /// Half-life for decaying modes (positive), doubling time for growing (negative).
    pub half_life: Option<f64>,
    /// Stability classification.
    pub stability: Stability,
    /// Mode amplitude |b|.
    pub amplitude: f64,
}

/// Stability classification of a mode or system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stability {
    Decaying,
    Neutral,
    Growing,
}

impl std::fmt::Display for Stability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stability::Decaying => write!(f, "decaying"),
            Stability::Neutral => write!(f, "neutral"),
            Stability::Growing => write!(f, "growing"),
        }
    }
}

/// Result of stability analysis.
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Whether all modes are decaying or neutral.
    pub is_stable: bool,
    /// Whether any mode is growing.
    pub is_unstable: bool,
    /// Whether any mode is exactly neutral (within tolerance).
    pub is_marginal: bool,
    /// Maximum eigenvalue magnitude.
    pub spectral_radius: f64,
    /// Per-mode stability classification.
    pub mode_stability: Vec<Stability>,
}

/// Error metrics for reconstruction quality.
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    /// Root mean square error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Mean absolute percentage error.
    pub mape: f64,
    /// Relative error (Frobenius norm ratio).
    pub relative_error: f64,
    /// Per-variable RMSE.
    pub per_variable_rmse: Vec<f64>,
}

/// Criterion for selecting dominant modes.
#[derive(Debug, Clone, Copy)]
pub enum DominantCriterion {
    /// Sort by amplitude |b|.
    Amplitude,
    /// Sort by energy |b| × |λ|.
    Energy,
    /// Sort by stability (closest to unit circle first).
    Stability,
}

/// Result of residual analysis.
#[derive(Debug, Clone)]
pub struct ResidualResult {
    /// Overall residual Frobenius norm.
    pub residual_norm: f64,
    /// Relative residual (residual_norm / data_norm).
    pub residual_relative: f64,
    /// Per-step residual norms.
    pub per_step_residual: Vec<f64>,
    /// Per-mode residual contributions.
    pub per_mode_residual: Vec<f64>,
}

/// Result of pseudospectrum computation.
#[derive(Debug, Clone)]
pub struct PseudospectrumResult {
    /// Real axis grid points.
    pub x: Vec<f64>,
    /// Imaginary axis grid points.
    pub y: Vec<f64>,
    /// Minimum singular values at each grid point (grid_n × grid_n, row-major).
    pub sigma_min: Vec<Vec<f64>>,
    /// DMD eigenvalues for reference.
    pub eigenvalues: Vec<C64>,
    /// Epsilon contour levels.
    pub epsilon: Vec<f64>,
}

/// Result of convergence analysis.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Sample sizes used.
    pub sample_sizes: Vec<usize>,
    /// Eigenvalues at each sample size.
    pub eigenvalues: Vec<Vec<C64>>,
    /// Max eigenvalue magnitude changes between successive fits.
    pub eigenvalue_changes: Vec<f64>,
    /// Estimated convergence rate (O(1/m^alpha)), None if insufficient data.
    pub convergence_estimate: Option<f64>,
}
