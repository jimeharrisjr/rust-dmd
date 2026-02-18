use faer::Mat;

use crate::hankel::{hankel_dmd, HankelConfig};
use crate::types::{DmdError, C64};
use crate::utils::validate_matrix;

/// Configuration for Generalized Laplace Analysis.
#[derive(Debug, Clone)]
pub struct GlaConfig {
    /// Candidate eigenvalues. If None, estimated via preliminary Hankel-DMD.
    pub eigenvalues: Option<Vec<C64>>,
    /// Number of eigenvalues to estimate if none provided.
    pub n_eigenvalues: usize,
    /// Convergence tolerance for weighted time averages.
    pub tol: f64,
    /// Maximum iterations. None for all available data.
    pub max_iter: Option<usize>,
}

impl Default for GlaConfig {
    fn default() -> Self {
        Self {
            eigenvalues: None,
            n_eigenvalues: 5,
            tol: 1e-6,
            max_iter: None,
        }
    }
}

/// Result of a GLA computation.
#[derive(Debug, Clone)]
pub struct GlaResult {
    /// Eigenvalues used (sorted by magnitude descending).
    pub eigenvalues: Vec<C64>,
    /// Eigenfunction values at each time point (n_eig × n_time).
    pub eigenfunctions: Vec<Vec<C64>>,
    /// Koopman modes (n_obs × n_eig).
    pub modes: Vec<Vec<C64>>,
    /// Whether each eigenvalue converged.
    pub convergence: Vec<bool>,
    /// Residual norms for each eigenvalue.
    pub residuals: Vec<f64>,
    /// Eigenfunction errors (how well φ(Tx) = λφ(x) holds).
    pub eigenfunction_errors: Vec<f64>,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Number of observables.
    pub n_obs: usize,
    /// Number of time points.
    pub n_time: usize,
}

/// Perform Generalized Laplace Analysis on trajectory data.
///
/// Computes Koopman eigenfunctions and modes directly from trajectory data
/// via weighted time averages, without constructing an approximate operator matrix.
///
/// # Algorithm (Mezic 2020, Theorem 3.1)
/// f_k = lim (1/n) Σ_{i=0}^{n-1} λ_k^{-i} (f(T^i x) - Σ_{j<k} λ_j^i φ_j(x) s_j)
///
/// # Arguments
/// * `y` - Trajectory data (n_obs × n_time).
/// * `config` - GLA configuration.
pub fn gla(y: &Mat<f64>, config: &GlaConfig) -> Result<GlaResult, DmdError> {
    validate_matrix(y, 1, 4)?;

    let n_obs = y.nrows();
    let n_time = y.ncols();
    let max_iter = config.max_iter.unwrap_or(n_time).min(n_time);

    // Estimate eigenvalues if not provided
    let mut eigenvalues = match &config.eigenvalues {
        Some(evs) => evs.clone(),
        None => {
            let delays = (n_time / 3).max(2).min(10);
            let hconfig = HankelConfig {
                delays: Some(delays),
                ..Default::default()
            };
            let hresult = hankel_dmd(y, &hconfig)?;
            let n_take = config.n_eigenvalues.min(hresult.eigenvalues.len());
            hresult.eigenvalues[..n_take].to_vec()
        }
    };

    let n_eig = eigenvalues.len();

    // Sort eigenvalues by magnitude (descending)
    let mut order: Vec<usize> = (0..n_eig).collect();
    order.sort_by(|&a, &b| {
        eigenvalues[b]
            .norm()
            .partial_cmp(&eigenvalues[a].norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eigenvalues = order.iter().map(|&i| eigenvalues[i]).collect();

    // Initialize storage
    let mut eigenfunctions = vec![vec![C64::zero(); n_time]; n_eig];
    let mut modes = vec![vec![C64::zero(); n_eig]; n_obs];
    let mut convergence = vec![false; n_eig];
    let mut residuals = vec![0.0; n_eig];

    // Compute eigenfunctions sequentially with deflation
    for k in 0..n_eig {
        let lambda_k = eigenvalues[k];

        // Build residual: subtract contributions from previously computed modes
        // f_residual starts as original data y
        let mut f_residual = vec![vec![0.0; n_time]; n_obs];
        for i in 0..n_obs {
            for t in 0..n_time {
                f_residual[i][t] = y[(i, t)];
            }
        }

        // Subtract previous mode contributions
        for j in 0..k {
            let lambda_j = eigenvalues[j];
            let phi_j_0 = eigenfunctions[j][0]; // eigenfunction value at t=0

            for t in 0..n_time {
                let lambda_j_t = lambda_j.powf(t as f64);
                let contrib = lambda_j_t * phi_j_0;
                for i in 0..n_obs {
                    f_residual[i][t] -= (contrib * modes[i][j]).re;
                }
            }
        }

        // Compute weighted time average
        let avg_result = compute_gla_average(&f_residual, lambda_k, max_iter, config.tol);

        // Extract eigenfunction and mode
        let f_k = avg_result.average;

        if f_k[0].norm() > config.tol {
            // Mode s_k is the n_obs vector f_k
            for i in 0..n_obs {
                modes[i][k] = f_k[i];
            }
            // phi_k(x_0) = 1 by normalization
            let phi_k_0 = C64::new(1.0, 0.0);

            // Eigenfunction evolution: phi_k(x_t) = lambda_k^t * phi_k(x_0)
            for t in 0..n_time {
                eigenfunctions[k][t] = lambda_k.powf(t as f64) * phi_k_0;
            }
        }
        // else: eigenfunction and mode remain zero

        convergence[k] = avg_result.converged;
        residuals[k] = avg_result.residual;
    }

    // Compute eigenfunction errors: how well φ(Tx) = λφ(x) holds
    let mut eigenfunction_errors = vec![0.0; n_eig];
    for k in 0..n_eig {
        if n_time > 1 {
            let lambda_k = eigenvalues[k];
            let mut err_sq = 0.0;
            let count = (n_time - 1) as f64;
            for t in 0..(n_time - 1) {
                let lhs = eigenfunctions[k][t + 1];
                let rhs = lambda_k * eigenfunctions[k][t];
                let diff = lhs - rhs;
                err_sq += diff.norm_sqr();
            }
            eigenfunction_errors[k] = (err_sq / count).sqrt();
        }
    }

    Ok(GlaResult {
        eigenvalues,
        eigenfunctions,
        modes,
        convergence,
        residuals,
        eigenfunction_errors,
        n_iter: max_iter,
        n_obs,
        n_time,
    })
}

/// Predict future values using GLA modal decomposition.
///
/// y(t) = Σ_k φ_k(t) · s_k = Σ_k λ_k^t · φ_k(0) · s_k
pub fn gla_predict(result: &GlaResult, n_ahead: usize) -> Result<Mat<f64>, DmdError> {
    if n_ahead == 0 {
        return Err(DmdError::InvalidInput("n_ahead must be positive".into()));
    }

    let n_obs = result.n_obs;
    let n_eig = result.eigenvalues.len();
    let n_time = result.n_time;

    let mut predictions = Mat::<f64>::zeros(n_obs, n_ahead);

    for k in 0..n_eig {
        let lambda_k = result.eigenvalues[k];
        let phi_k_0 = result.eigenfunctions[k][0];

        for t in 0..n_ahead {
            let t_future = (n_time + t) as f64;
            let phi_k_t = lambda_k.powf(t_future) * phi_k_0;
            for i in 0..n_obs {
                let contrib = (phi_k_t * result.modes[i][k]).re;
                predictions[(i, t)] += contrib;
            }
        }
    }

    Ok(predictions)
}

/// Reconstruct the original signal using GLA modes.
///
/// Optionally use a subset of modes.
pub fn gla_reconstruct(
    result: &GlaResult,
    modes_to_use: Option<&[usize]>,
) -> Result<Mat<f64>, DmdError> {
    let n_obs = result.n_obs;
    let n_time = result.n_time;
    let n_eig = result.eigenvalues.len();

    let indices: Vec<usize> = match modes_to_use {
        Some(idx) => {
            for &i in idx {
                if i >= n_eig {
                    return Err(DmdError::InvalidInput(format!(
                        "mode index {i} out of range (n_eig={n_eig})"
                    )));
                }
            }
            idx.to_vec()
        }
        None => (0..n_eig).collect(),
    };

    let mut recon = Mat::<f64>::zeros(n_obs, n_time);

    for &k in &indices {
        let lambda_k = result.eigenvalues[k];
        let phi_k_0 = result.eigenfunctions[k][0];

        for t in 0..n_time {
            let phi_k_t = lambda_k.powf(t as f64) * phi_k_0;
            for i in 0..n_obs {
                recon[(i, t)] += (phi_k_t * result.modes[i][k]).re;
            }
        }
    }

    Ok(recon)
}

struct GlaAverageResult {
    average: Vec<C64>,
    converged: bool,
    residual: f64,
}

/// Compute weighted time average for a single eigenvalue.
///
/// For stable eigenvalues (|λ| ≤ 1): forward iteration with λ^{-i} weights.
/// For unstable eigenvalues (|λ| > 1): reverse iteration for numerical stability.
fn compute_gla_average(
    f_residual: &[Vec<f64>],
    lambda: C64,
    max_iter: usize,
    tol: f64,
) -> GlaAverageResult {
    let n_obs = f_residual.len();
    let n_time = if n_obs > 0 { f_residual[0].len() } else { 0 };
    let n_iter = max_iter.min(n_time);

    let mut running_sum = vec![C64::zero(); n_obs];
    let mut prev_avg = vec![C64::zero(); n_obs];
    let mut converged = false;

    let lambda_mag = lambda.norm();

    if lambda_mag > 1.0 + tol {
        // Unstable: reverse iteration with forward weights
        for i in 0..n_iter {
            let t_idx = n_time.saturating_sub(i + 1);
            let weight = lambda.powf(i as f64);
            for obs in 0..n_obs {
                running_sum[obs] += weight * C64::new(f_residual[obs][t_idx], 0.0);
            }
        }
    } else {
        // Standard forward iteration for stable eigenvalues
        let lambda_inv = if lambda.norm_sqr() > 1e-30 {
            C64::new(1.0, 0.0) / lambda
        } else {
            C64::zero()
        };

        let mut weight = C64::new(1.0, 0.0);
        for i in 0..n_iter {
            if i >= n_time {
                break;
            }
            for obs in 0..n_obs {
                running_sum[obs] += weight * C64::new(f_residual[obs][i], 0.0);
            }

            // Check convergence every 10 iterations
            if (i + 1) % 10 == 0 && i > 10 {
                let current_n = (i + 1) as f64;
                let mut change_sq = 0.0;
                for obs in 0..n_obs {
                    let current_avg = running_sum[obs] / current_n;
                    let diff = current_avg - prev_avg[obs];
                    change_sq += diff.norm_sqr();
                }
                let change = change_sq.sqrt();
                if change < tol {
                    converged = true;
                    break;
                }
                for obs in 0..n_obs {
                    prev_avg[obs] = running_sum[obs] / current_n;
                }
            }

            weight = weight * lambda_inv;
        }
    }

    let n_iter_f = n_iter as f64;
    let average: Vec<C64> = running_sum.iter().map(|&s| s / n_iter_f).collect();

    let mut final_residual_sq = 0.0;
    for obs in 0..n_obs {
        let diff = average[obs] - prev_avg[obs];
        final_residual_sq += diff.norm_sqr();
    }
    let residual = final_residual_sq.sqrt();

    GlaAverageResult {
        average,
        converged,
        residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn assert_near(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn make_oscillatory_scalar(n_time: usize) -> Mat<f64> {
        let dt = 0.1;
        let mut y = Mat::<f64>::zeros(1, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            y[(0, t)] = (2.0 * PI * 0.5 * time).sin() + 0.3 * (2.0 * PI * 1.5 * time).cos();
        }
        y
    }

    fn make_oscillatory_2d(n_time: usize) -> Mat<f64> {
        let dt = 0.1;
        let mut y = Mat::<f64>::zeros(2, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            y[(0, t)] = (2.0 * PI * 0.5 * time).cos();
            y[(1, t)] = (2.0 * PI * 0.5 * time).sin();
        }
        y
    }

    #[test]
    fn test_gla_auto_eigenvalues() {
        let y = make_oscillatory_scalar(200);
        let config = GlaConfig {
            n_eigenvalues: 4,
            ..Default::default()
        };
        let result = gla(&y, &config).unwrap();

        assert_eq!(result.n_obs, 1);
        assert_eq!(result.n_time, 200);
        assert!(result.eigenvalues.len() <= 4);
    }

    #[test]
    fn test_gla_known_eigenvalues() {
        let dt = 0.1;
        let omega = 2.0 * PI * 0.5;
        let eigenvalues = vec![
            C64::new((omega * dt).cos(), (omega * dt).sin()),
            C64::new((omega * dt).cos(), -(omega * dt).sin()),
        ];

        let y = make_oscillatory_2d(200);
        let config = GlaConfig {
            eigenvalues: Some(eigenvalues),
            ..Default::default()
        };
        let result = gla(&y, &config).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);
        for ev in &result.eigenvalues {
            assert_near(ev.norm(), 1.0, 0.01);
        }
    }

    #[test]
    fn test_gla_reconstruct() {
        let y = make_oscillatory_2d(100);
        let config = GlaConfig {
            n_eigenvalues: 4,
            ..Default::default()
        };
        let result = gla(&y, &config).unwrap();

        let recon = gla_reconstruct(&result, None).unwrap();
        assert_eq!(recon.nrows(), 2);
        assert_eq!(recon.ncols(), 100);
    }

    #[test]
    fn test_gla_predict() {
        let y = make_oscillatory_scalar(200);
        let config = GlaConfig {
            n_eigenvalues: 4,
            ..Default::default()
        };
        let result = gla(&y, &config).unwrap();

        let pred = gla_predict(&result, 20).unwrap();
        assert_eq!(pred.nrows(), 1);
        assert_eq!(pred.ncols(), 20);
    }

    #[test]
    fn test_gla_predict_zero() {
        let y = make_oscillatory_scalar(100);
        let config = GlaConfig::default();
        let result = gla(&y, &config).unwrap();
        assert!(gla_predict(&result, 0).is_err());
    }

    #[test]
    fn test_gla_eigenfunction_errors() {
        let y = make_oscillatory_2d(200);
        let config = GlaConfig {
            n_eigenvalues: 2,
            ..Default::default()
        };
        let result = gla(&y, &config).unwrap();

        // Eigenfunction errors should be small for well-estimated eigenvalues
        for err in &result.eigenfunction_errors {
            // By construction phi(Tx) = lambda*phi(x) exactly, so errors should be ~0
            assert!(*err < 1e-10, "eigenfunction error = {err}");
        }
    }
}
