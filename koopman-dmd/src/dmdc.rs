//! Dynamic Mode Decomposition with control (DMDc).
//!
//! Identifies the forced linear system
//!
//! ```text
//! x_{t+1} ≈ A x_t + B u_t
//! ```
//!
//! from snapshot pairs `(x1, x2)` and control inputs `u`, following Proctor,
//! Brunton & Kutz (2016), *SIAM J. Applied Dynamical Systems* 15(1), 142-161.
//!
//! Unlike [`dmd()`](crate::dmd()), which takes one contiguous trajectory,
//! `dmdc` takes **explicit pair matrices**: `x1` holds states at time `t`,
//! `x2` the states one step later, and `u` the control input applied during
//! each transition. Columns may therefore come from many concatenated
//! trajectories, and pairs may be freely masked out by the caller.
//!
//! Two identification modes:
//!
//! - **Unknown B** (`known_b: None`): jointly solves `[A B] = X₂ Ω⁺` with
//!   `Ω = [X₁; U]` via truncated SVD. Requires the control input to be
//!   persistently exciting and *exogenous* — identifying both `A` and `B`
//!   from closed-loop (state-feedback) data is biased and non-unique.
//! - **Known B** (`known_b: Some(B)`): subtracts the known input response and
//!   solves only `A = (X₂ − B U) X₁⁺`. Preferred whenever the input coupling
//!   is known by construction; immune to the closed-loop caveat above.
//!
//! Results are returned as real `faer::Mat<f64>` matrices, ready for use in
//! stepping loops without conversion.

use faer::Mat;

use crate::types::{DmdError, SvdComponents, C64};
use crate::utils::{determine_rank, validate_matrix};

/// Relative singular-value floor: an explicitly requested rank whose smallest
/// retained singular value falls below `σ_max · RANK_FLOOR` exceeds the
/// numerical rank of the data.
const RANK_FLOOR: f64 = 1e-12;

/// Configuration for DMDc.
#[derive(Debug, Clone)]
pub struct DmdcConfig {
    /// Truncation rank for the regression-input SVD (`[X₁; U]`, or `X₁` when
    /// `known_b` is set). `None` selects 99 % cumulative variance.
    pub rank_input: Option<usize>,
    /// Optional second projection: rank of the output basis (SVD of `X₂`).
    /// `Some(r)` produces the reduced operator pair `(Ã, B̃)` in an
    /// `r`-dimensional basis; `None` keeps everything full-order (the basis
    /// is the identity and `Ã = A`).
    pub rank_output: Option<usize>,
    /// Time step between snapshot pairs.
    pub dt: f64,
    /// Known input matrix `B` (n × q). When set, `B` is not estimated.
    pub known_b: Option<Mat<f64>>,
}

impl Default for DmdcConfig {
    fn default() -> Self {
        Self {
            rank_input: None,
            rank_output: None,
            dt: 1.0,
            known_b: None,
        }
    }
}

/// Result of a DMDc computation. All matrices are real.
#[derive(Debug, Clone)]
pub struct DmdcResult {
    /// State-transition matrix `A` (n × n).
    pub a: Mat<f64>,
    /// Input matrix `B` (n × q). A copy of `known_b` when that was supplied.
    pub b: Mat<f64>,
    /// Reduced operator `Ã = ÛᵀAÛ` (r × r); equals `A` when no output
    /// projection was requested.
    pub a_tilde: Mat<f64>,
    /// Reduced input matrix `B̃ = ÛᵀB` (r × q).
    pub b_tilde: Mat<f64>,
    /// Orthonormal output basis `Û` (n × r); identity when no output
    /// projection was requested.
    pub basis: Mat<f64>,
    /// Eigenvalues of `Ã`.
    pub eigenvalues: Vec<C64>,
    /// Truncated SVD of the regression input (`[X₁; U]` or `X₁`).
    pub svd_input: SvdComponents,
    /// Rank used for the regression-input SVD.
    pub rank_input: usize,
    /// Rank of the output basis (`n` when no projection was requested).
    pub rank_output: usize,
    /// Time step.
    pub dt: f64,
}

/// Identify `A` (and optionally `B`) from snapshot pairs.
///
/// # Arguments
/// * `x1` — states at time `t` (n × m).
/// * `x2` — states at time `t + 1` (n × m).
/// * `u` — control inputs during each transition (q × m). May have zero rows
///   for an autonomous multi-trajectory fit.
/// * `config` — see [`DmdcConfig`].
pub fn dmdc(
    x1: &Mat<f64>,
    x2: &Mat<f64>,
    u: &Mat<f64>,
    config: &DmdcConfig,
) -> Result<DmdcResult, DmdError> {
    validate_matrix(x1, 1, 2)?;
    let n = x1.nrows();
    let m = x1.ncols();
    let q = u.nrows();
    if x2.nrows() != n || x2.ncols() != m {
        return Err(DmdError::InvalidInput(format!(
            "x2 is {}×{}, expected {n}×{m} to match x1",
            x2.nrows(),
            x2.ncols()
        )));
    }
    if q > 0 && u.ncols() != m {
        return Err(DmdError::InvalidInput(format!(
            "u has {} columns, expected {m} to match the snapshot pairs",
            u.ncols()
        )));
    }
    if let Some(b) = &config.known_b {
        if b.nrows() != n || b.ncols() != q {
            return Err(DmdError::InvalidInput(format!(
                "known_b is {}×{}, expected {n}×{q}",
                b.nrows(),
                b.ncols()
            )));
        }
    }
    if config.dt <= 0.0 || !config.dt.is_finite() {
        return Err(DmdError::InvalidInput(format!(
            "dt must be positive and finite (dt = {})",
            config.dt
        )));
    }

    let (a, b, svd_input, rank_input) = match &config.known_b {
        Some(b_known) => {
            // Y = X₂ − B U, then A = Y X₁⁺ via truncated SVD of X₁.
            let mut y = x2.clone();
            if q > 0 {
                let bu = b_known * u;
                for j in 0..m {
                    for i in 0..n {
                        y[(i, j)] -= bu[(i, j)];
                    }
                }
            }
            let (g, svd, rank) = solve_via_svd(x1, &y, config.rank_input)?;
            (g, b_known.clone(), svd, rank)
        }
        None => {
            // Ω = [X₁; U], [A B] = X₂ Ω⁺.
            let mut omega = Mat::<f64>::zeros(n + q, m);
            for j in 0..m {
                for i in 0..n {
                    omega[(i, j)] = x1[(i, j)];
                }
                for i in 0..q {
                    omega[(n + i, j)] = u[(i, j)];
                }
            }
            let (g, svd, rank) = solve_via_svd(&omega, x2, config.rank_input)?;
            let a = g.as_ref().subcols(0, n).to_owned();
            let b = g.as_ref().subcols(n, q).to_owned();
            (a, b, svd, rank)
        }
    };

    // Optional output projection through the SVD of X₂.
    let (basis, a_tilde, b_tilde, rank_output) = match config.rank_output {
        None => {
            let mut eye = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                eye[(i, i)] = 1.0;
            }
            (eye, a.clone(), b.clone(), n)
        }
        Some(r_req) => {
            let svd = x2
                .svd()
                .map_err(|e| DmdError::SvdFailed(format!("{e:?}")))?;
            let s_col = svd.S().column_vector();
            let s_vals: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
            let r = determine_rank(&s_vals, Some(r_req), 0.99).min(n);
            check_rank_floor(&s_vals, r)?;
            let u_hat = svd.U().subcols(0, r).to_owned();
            let ut_a = u_hat.transpose() * &a;
            let a_tilde = &ut_a * &u_hat;
            let b_tilde = u_hat.transpose() * &b;
            (u_hat, a_tilde, b_tilde, r)
        }
    };

    let eigenvalues = eigenvalues_of(&a_tilde)?;

    Ok(DmdcResult {
        a,
        b,
        a_tilde,
        b_tilde,
        basis,
        eigenvalues,
        svd_input,
        rank_input,
        dt: config.dt,
        rank_output,
    })
}

/// Least-squares `G = Y · pinv(X)` through a truncated SVD of `X`:
/// `G = Y V Σ⁻¹ Uᵀ`. Returns `(G, truncated SVD of X, rank)`.
fn solve_via_svd(
    x: &Mat<f64>,
    y: &Mat<f64>,
    rank: Option<usize>,
) -> Result<(Mat<f64>, SvdComponents, usize), DmdError> {
    let svd = x.svd().map_err(|e| DmdError::SvdFailed(format!("{e:?}")))?;
    let s_col = svd.S().column_vector();
    let s_vals: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
    let r = determine_rank(&s_vals, rank, 0.99);
    check_rank_floor(&s_vals, r)?;

    let u = svd.U().subcols(0, r).to_owned();
    let v = svd.V().subcols(0, r).to_owned();
    let s: Vec<f64> = s_vals[..r].to_vec();

    // G = Y V Σ⁻¹ Uᵀ.
    let y_v = y * &v; // (rows(y) × r)
    let mut y_v_sinv = y_v;
    for j in 0..r {
        for i in 0..y_v_sinv.nrows() {
            y_v_sinv[(i, j)] /= s[j];
        }
    }
    let g = &y_v_sinv * u.transpose();

    Ok((g, SvdComponents { u, s, v }, r))
}

/// Reject explicitly requested ranks that exceed the numerical rank.
fn check_rank_floor(s_vals: &[f64], r: usize) -> Result<(), DmdError> {
    let s_max = s_vals.first().copied().unwrap_or(0.0);
    if r == 0 || s_max <= 0.0 {
        return Err(DmdError::InvalidInput("data matrix is zero".into()));
    }
    if s_vals[r - 1] < RANK_FLOOR * s_max {
        return Err(DmdError::InvalidInput(format!(
            "requested rank {r} exceeds the numerical rank of the data \
             (σ_{r} / σ_1 = {:.3e})",
            s_vals[r - 1] / s_max
        )));
    }
    Ok(())
}

/// Eigenvalues of a real square matrix.
fn eigenvalues_of(a: &Mat<f64>) -> Result<Vec<C64>, DmdError> {
    let eigen = a
        .as_ref()
        .eigen()
        .map_err(|e| DmdError::EigenFailed(format!("{e:?}")))?;
    let diag = eigen.S().column_vector();
    Ok((0..a.nrows())
        .map(|i| C64::new(diag[i].re, diag[i].im))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_mat_near(got: &Mat<f64>, want: &[&[f64]], tol: f64) {
        assert_eq!(got.nrows(), want.len());
        for (i, row) in want.iter().enumerate() {
            assert_eq!(got.ncols(), row.len());
            for (j, &w) in row.iter().enumerate() {
                assert!(
                    (got[(i, j)] - w).abs() <= tol,
                    "entry ({i}, {j}): got {}, want {w}",
                    got[(i, j)]
                );
            }
        }
    }

    /// Simulate x_{t+1} = A₀ x_t + B₀ u_t with a persistently exciting input,
    /// returning (x1, x2, u) pair matrices.
    fn simulate_forced(m: usize) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
        let a0 = [[0.9, 0.1], [0.0, 0.8]];
        let b0 = [0.5, 1.0];
        let mut x1 = Mat::<f64>::zeros(2, m);
        let mut x2 = Mat::<f64>::zeros(2, m);
        let mut u = Mat::<f64>::zeros(1, m);
        let mut x = [1.0, -0.5];
        for t in 0..m {
            let ut = (0.7 * t as f64).sin() + 0.5 * (2.3 * t as f64 + 1.0).cos();
            x1[(0, t)] = x[0];
            x1[(1, t)] = x[1];
            u[(0, t)] = ut;
            let next = [
                a0[0][0] * x[0] + a0[0][1] * x[1] + b0[0] * ut,
                a0[1][0] * x[0] + a0[1][1] * x[1] + b0[1] * ut,
            ];
            x2[(0, t)] = next[0];
            x2[(1, t)] = next[1];
            x = next;
        }
        (x1, x2, u)
    }

    #[test]
    fn recovers_a_and_b_jointly() {
        let (x1, x2, u) = simulate_forced(120);
        let config = DmdcConfig {
            rank_input: Some(3),
            ..Default::default()
        };
        let res = dmdc(&x1, &x2, &u, &config).unwrap();
        assert_mat_near(&res.a, &[&[0.9, 0.1], &[0.0, 0.8]], 1e-9);
        assert_mat_near(&res.b, &[&[0.5], &[1.0]], 1e-9);
        assert_eq!(res.rank_input, 3);
        assert_eq!(res.rank_output, 2);
        // Full-order default: basis is the identity and Ã = A.
        assert_mat_near(&res.basis, &[&[1.0, 0.0], &[0.0, 1.0]], 0.0);
        // Eigenvalues of the recovered A are the true ones.
        let mut mags: Vec<f64> = res.eigenvalues.iter().map(|e| e.norm()).collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((mags[0] - 0.8).abs() < 1e-9);
        assert!((mags[1] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn known_b_pins_the_input_matrix() {
        let (x1, x2, u) = simulate_forced(120);
        let mut b_known = Mat::<f64>::zeros(2, 1);
        b_known[(0, 0)] = 0.5;
        b_known[(1, 0)] = 1.0;
        let config = DmdcConfig {
            rank_input: Some(2),
            known_b: Some(b_known),
            ..Default::default()
        };
        let res = dmdc(&x1, &x2, &u, &config).unwrap();
        assert_mat_near(&res.a, &[&[0.9, 0.1], &[0.0, 0.8]], 1e-9);
        assert_mat_near(&res.b, &[&[0.5], &[1.0]], 0.0);
    }

    #[test]
    fn autonomous_pairs_fit_with_zero_control_rows() {
        // The q = 0 path: multi-trajectory autonomous identification from
        // explicit pairs, which dmd() cannot do.
        let a0 = [[0.95, 0.02], [0.0, 0.85]];
        let mut x1 = Mat::<f64>::zeros(2, 80);
        let mut x2 = Mat::<f64>::zeros(2, 80);
        // Two trajectories from different initial conditions, concatenated.
        let mut col = 0;
        for start in [[1.0, 0.5], [-0.3, 1.2]] {
            let mut x = start;
            for _ in 0..40 {
                x1[(0, col)] = x[0];
                x1[(1, col)] = x[1];
                let next = [a0[0][0] * x[0] + a0[0][1] * x[1], a0[1][1] * x[1]];
                x2[(0, col)] = next[0];
                x2[(1, col)] = next[1];
                x = next;
                col += 1;
            }
        }
        let u = Mat::<f64>::zeros(0, 80);
        let config = DmdcConfig {
            rank_input: Some(2),
            ..Default::default()
        };
        let res = dmdc(&x1, &x2, &u, &config).unwrap();
        assert_mat_near(&res.a, &[&[0.95, 0.02], &[0.0, 0.85]], 1e-9);
        assert_eq!(res.b.ncols(), 0);
    }

    #[test]
    fn output_projection_produces_reduced_operators() {
        let (x1, x2, u) = simulate_forced(120);
        let config = DmdcConfig {
            rank_input: Some(3),
            rank_output: Some(2),
            ..Default::default()
        };
        let res = dmdc(&x1, &x2, &u, &config).unwrap();
        assert_eq!(res.a_tilde.nrows(), 2);
        assert_eq!(res.b_tilde.nrows(), 2);
        assert_eq!(res.basis.ncols(), 2);
        // Basis columns are orthonormal.
        for i in 0..2 {
            for j in 0..2 {
                let dot: f64 = (0..2).map(|k| res.basis[(k, i)] * res.basis[(k, j)]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12);
            }
        }
        // With r = n the projection is a similarity transform: eigenvalues of
        // Ã match those of A.
        let mut mags: Vec<f64> = res.eigenvalues.iter().map(|e| e.norm()).collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((mags[0] - 0.8).abs() < 1e-9);
        assert!((mags[1] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn dimension_mismatches_are_rejected() {
        let (x1, x2, u) = simulate_forced(50);
        let config = DmdcConfig::default();
        // x2 shape mismatch.
        let bad_x2 = Mat::<f64>::zeros(2, 49);
        assert!(dmdc(&x1, &bad_x2, &u, &config).is_err());
        // u column mismatch.
        let bad_u = Mat::<f64>::zeros(1, 49);
        assert!(dmdc(&x1, &x2, &bad_u, &config).is_err());
        // known_b shape mismatch.
        let config_bad_b = DmdcConfig {
            known_b: Some(Mat::<f64>::zeros(3, 1)),
            ..Default::default()
        };
        assert!(dmdc(&x1, &x2, &u, &config_bad_b).is_err());
        // Nonpositive dt.
        let config_bad_dt = DmdcConfig {
            dt: 0.0,
            ..Default::default()
        };
        assert!(dmdc(&x1, &x2, &u, &config_bad_dt).is_err());
    }

    #[test]
    fn rank_beyond_numerical_rank_is_rejected() {
        // Rank-1 data: x1 columns are all multiples of one vector.
        let mut x1 = Mat::<f64>::zeros(2, 10);
        let mut x2 = Mat::<f64>::zeros(2, 10);
        for t in 0..10 {
            let s = 0.9f64.powi(t as i32);
            x1[(0, t)] = s;
            x1[(1, t)] = 2.0 * s;
            x2[(0, t)] = 0.9 * s;
            x2[(1, t)] = 1.8 * s;
        }
        let u = Mat::<f64>::zeros(0, 10);
        let config = DmdcConfig {
            rank_input: Some(2),
            ..Default::default()
        };
        assert!(dmdc(&x1, &x2, &u, &config).is_err());
    }
}
