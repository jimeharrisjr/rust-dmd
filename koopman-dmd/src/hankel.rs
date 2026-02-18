use faer::Mat;

use crate::types::{DmdError, SvdComponents, C64};
use crate::utils::{determine_rank, validate_matrix};

/// Configuration for Hankel-DMD.
#[derive(Debug, Clone)]
pub struct HankelConfig {
    /// Number of time delays. None for automatic (n_time / 3).
    pub delays: Option<usize>,
    /// Truncation rank. None for automatic selection.
    pub rank: Option<usize>,
    /// Time step between snapshots.
    pub dt: f64,
}

impl Default for HankelConfig {
    fn default() -> Self {
        Self {
            delays: None,
            rank: None,
            dt: 1.0,
        }
    }
}

/// Result of a Hankel-DMD computation.
#[derive(Debug, Clone)]
pub struct HankelDmdResult {
    /// DMD modes Φ in delay-embedded space (n_delays+1 × r).
    pub modes: Vec<Vec<C64>>,
    /// Eigenvalues λ (r).
    pub eigenvalues: Vec<C64>,
    /// Initial amplitudes b (r).
    pub amplitudes: Vec<C64>,
    /// Truncation rank used.
    pub rank: usize,
    /// Truncated SVD components.
    pub svd: SvdComponents,
    /// Reduced DMD matrix Ã (r × r).
    pub a_tilde: Vec<Vec<C64>>,
    /// Number of delays used.
    pub delays: usize,
    /// Number of observables in original data.
    pub n_obs: usize,
    /// Number of time points in original data.
    pub n_time_original: usize,
    /// Hankel matrix dimensions (rows, cols).
    pub hankel_dim: (usize, usize),
    /// Companion matrix (delays+1 × delays+1), real.
    pub companion: Mat<f64>,
    /// Residual norm of the Hankel-DMD fit.
    pub residual: f64,
    /// Time step.
    pub dt: f64,
}

/// Build a Hankel-Takens matrix from time series data.
///
/// For a multivariate time series y (n_obs × n_time) with `delays` delays,
/// the Hankel matrix has dimensions ((delays+1)*n_obs × (n_time - delays)).
///
/// H[d*n_obs + i, t] = y[i, t + d] for d = 0..delays, i = 0..n_obs, t = 0..n_time-delays
pub fn build_hankel_matrix(y: &Mat<f64>, delays: usize) -> Result<Mat<f64>, DmdError> {
    let n_obs = y.nrows();
    let n_time = y.ncols();

    if n_time <= delays {
        return Err(DmdError::InvalidInput(format!(
            "not enough time points ({n_time}) for {delays} delays"
        )));
    }

    let n_cols = n_time - delays;
    let n_rows = (delays + 1) * n_obs;
    let mut h = Mat::<f64>::zeros(n_rows, n_cols);

    for d in 0..=delays {
        for i in 0..n_obs {
            for t in 0..n_cols {
                h[(d * n_obs + i, t)] = y[(i, t + d)];
            }
        }
    }

    Ok(h)
}

/// Perform Hankel-DMD on time series data.
///
/// Uses time-delayed observables to form a Krylov subspace approximation
/// of the Koopman operator. Avoids the curse of dimensionality by using
/// the number of delays as the basis size.
///
/// # Arguments
/// * `y` - Time series data (n_obs × n_time). Can be a scalar (1 × n_time).
/// * `config` - Hankel-DMD configuration.
pub fn hankel_dmd(y: &Mat<f64>, config: &HankelConfig) -> Result<HankelDmdResult, DmdError> {
    validate_matrix(y, 1, 4)?;

    let n_obs = y.nrows();
    let n_time = y.ncols();

    // Determine number of delays
    let delays = config.delays.unwrap_or_else(|| {
        let d = n_time / 3;
        d.max(2).min(n_time - 2)
    });

    if n_time <= delays + 1 {
        return Err(DmdError::InvalidInput(format!(
            "not enough time points ({n_time}) for {delays} delays (need at least {})",
            delays + 2
        )));
    }

    // Build Hankel matrix
    let h = build_hankel_matrix(y, delays)?;
    let h_rows = h.nrows();
    let h_cols = h.ncols();

    // Split: H1 = H[:, 0..n-1], H2 = H[:, 1..n]
    let h1 = h.subcols(0, h_cols - 1).to_owned();
    let h2 = h.subcols(1, h_cols - 1).to_owned();

    // SVD of H1
    let svd = h1
        .svd()
        .map_err(|e| DmdError::SvdFailed(format!("{e:?}")))?;
    let u_full = svd.U();
    let v_full = svd.V();
    let s_col = svd.S().column_vector();

    let n_sv = s_col.nrows();
    let s_vals: Vec<f64> = (0..n_sv).map(|i| s_col[i]).collect();

    let rank = determine_rank(&s_vals, config.rank, 0.99);

    // Truncate
    let u = u_full.subcols(0, rank).to_owned();
    let v = v_full.subcols(0, rank).to_owned();
    let s: Vec<f64> = s_vals[..rank].to_vec();

    // Reduced operator: Ã = Uᵀ H₂ V Σ⁻¹
    let ut_h2 = u.transpose() * &h2;
    let ut_h2_v = &ut_h2 * &v;

    let mut a_tilde_real = Mat::<f64>::zeros(rank, rank);
    for i in 0..rank {
        for j in 0..rank {
            a_tilde_real[(i, j)] = ut_h2_v[(i, j)] / s[j];
        }
    }

    // Eigendecomposition
    let eigen = a_tilde_real
        .as_ref()
        .eigen()
        .map_err(|e| DmdError::EigenFailed(format!("{e:?}")))?;

    let ev_diag = eigen.S().column_vector();
    let ev_vecs = eigen.U();

    let mut eigenvalues = Vec::with_capacity(rank);
    let mut w_re = Mat::<f64>::zeros(rank, rank);
    let mut w_im = Mat::<f64>::zeros(rank, rank);

    for j in 0..rank {
        let ev = ev_diag[j];
        eigenvalues.push(C64::new(ev.re, ev.im));
        for i in 0..rank {
            let v_ij = ev_vecs[(i, j)];
            w_re[(i, j)] = v_ij.re;
            w_im[(i, j)] = v_ij.im;
        }
    }

    // DMD modes: Φ = H₂ V Σ⁻¹ W
    let h2_v = &h2 * &v;
    let mut h2_v_sinv = Mat::<f64>::zeros(h_rows, rank);
    for i in 0..h_rows {
        for j in 0..rank {
            h2_v_sinv[(i, j)] = h2_v[(i, j)] / s[j];
        }
    }

    let modes_re = &h2_v_sinv * &w_re;
    let modes_im = &h2_v_sinv * &w_im;

    let mut modes = vec![vec![C64::zero(); rank]; h_rows];
    for i in 0..h_rows {
        for j in 0..rank {
            modes[i][j] = C64::new(modes_re[(i, j)], modes_im[(i, j)]);
        }
    }

    // Amplitudes: b = Φ⁺ h₁_first
    let h1_first: Vec<f64> = (0..h_rows).map(|i| h1[(i, 0)]).collect();
    let amplitudes = solve_amplitudes_complex(&modes, &h1_first, h_rows, rank)?;

    // Companion matrix (from A_tilde projected back)
    let companion = a_tilde_real.clone();

    // Residual: ||H₂ - Ã_full H₁||_F / ||H₂||_F
    // Compute one-step prediction error in reduced space
    let h2_pred = {
        let a_full_approx = &u * &a_tilde_real * u.transpose();
        &a_full_approx * &h1
    };
    let mut residual_sq = 0.0;
    let mut h2_norm_sq = 0.0;
    for j in 0..(h_cols - 1) {
        for i in 0..h_rows {
            let diff = h2_pred[(i, j)] - h2[(i, j)];
            residual_sq += diff * diff;
            h2_norm_sq += h2[(i, j)] * h2[(i, j)];
        }
    }
    let residual = if h2_norm_sq > 0.0 {
        (residual_sq / h2_norm_sq).sqrt()
    } else {
        0.0
    };

    // A_tilde as complex
    let mut a_tilde_c = vec![vec![C64::zero(); rank]; rank];
    for i in 0..rank {
        for j in 0..rank {
            a_tilde_c[i][j] = C64::new(a_tilde_real[(i, j)], 0.0);
        }
    }

    let svd_components = SvdComponents { u, s, v };

    Ok(HankelDmdResult {
        modes,
        eigenvalues,
        amplitudes,
        rank,
        svd: svd_components,
        a_tilde: a_tilde_c,
        delays,
        n_obs,
        n_time_original: n_time,
        hankel_dim: (h_rows, h_cols),
        companion,
        residual,
        dt: config.dt,
    })
}

/// Reconstruct the original time series from Hankel-DMD.
///
/// Returns a matrix (n_obs × n_steps) by extracting the first n_obs rows
/// of the reconstructed Hankel vectors.
pub fn hankel_reconstruct(result: &HankelDmdResult, n_steps: usize) -> Result<Mat<f64>, DmdError> {
    let n_obs = result.n_obs;
    let h_rows = result.hankel_dim.0;
    let rank = result.rank;

    let mut recon = Mat::<f64>::zeros(n_obs, n_steps);

    for k in 0..n_steps {
        // Reconstruct the Hankel vector at step k
        // h(k) = Σ φ_j * b_j * λ_j^k
        let mut h_vec = vec![C64::zero(); h_rows];
        for j in 0..rank {
            let lambda_k = result.eigenvalues[j].powf(k as f64);
            let coeff = result.amplitudes[j] * lambda_k;
            for i in 0..h_rows {
                h_vec[i] += result.modes[i][j] * coeff;
            }
        }

        // Extract the first n_obs rows (current time, no delay)
        for i in 0..n_obs {
            recon[(i, k)] = h_vec[i].re;
        }
    }

    Ok(recon)
}

/// Predict future values using Hankel-DMD mode evolution.
///
/// Returns (n_obs × n_ahead) matrix of predicted values.
pub fn hankel_predict(result: &HankelDmdResult, n_ahead: usize) -> Result<Mat<f64>, DmdError> {
    if n_ahead == 0 {
        return Err(DmdError::InvalidInput("n_ahead must be positive".into()));
    }

    let n_obs = result.n_obs;
    let h_rows = result.hankel_dim.0;
    let rank = result.rank;
    let n_hankel_cols = result.hankel_dim.1;

    // Predict from the last Hankel column forward
    let start_k = n_hankel_cols; // first prediction step

    let mut predictions = Mat::<f64>::zeros(n_obs, n_ahead);

    for k in 0..n_ahead {
        let step = (start_k + k) as f64;
        let mut h_vec = vec![C64::zero(); h_rows];
        for j in 0..rank {
            let lambda_k = result.eigenvalues[j].powf(step);
            let coeff = result.amplitudes[j] * lambda_k;
            for i in 0..h_rows {
                h_vec[i] += result.modes[i][j] * coeff;
            }
        }

        for i in 0..n_obs {
            predictions[(i, k)] = h_vec[i].re;
        }
    }

    Ok(predictions)
}

/// Solve for complex amplitudes via least-squares: Φ b ≈ x₀.
fn solve_amplitudes_complex(
    modes: &[Vec<C64>],
    x0: &[f64],
    n_vars: usize,
    rank: usize,
) -> Result<Vec<C64>, DmdError> {
    let mut gram = vec![vec![C64::zero(); rank]; rank];
    let mut rhs = vec![C64::zero(); rank];

    for i in 0..rank {
        for j in 0..rank {
            let mut val = C64::zero();
            for k in 0..n_vars {
                val += modes[k][i].conj() * modes[k][j];
            }
            gram[i][j] = val;
        }
        let mut val = C64::zero();
        for k in 0..n_vars {
            val += modes[k][i].conj() * C64::new(x0[k], 0.0);
        }
        rhs[i] = val;
    }

    complex_gauss_solve(&gram, &rhs)
}

/// Solve complex system via Gaussian elimination with partial pivoting.
fn complex_gauss_solve(a: &[Vec<C64>], b: &[C64]) -> Result<Vec<C64>, DmdError> {
    let n = b.len();
    let mut aug: Vec<Vec<C64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        let mut max_norm = aug[col][col].norm();
        let mut max_row = col;
        for row in (col + 1)..n {
            let norm = aug[row][col].norm();
            if norm > max_norm {
                max_norm = norm;
                max_row = row;
            }
        }
        if max_norm < 1e-14 {
            return Err(DmdError::SolveFailed(
                "singular matrix in Hankel solve".into(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let sub = factor * aug[col][j];
                aug[row][j] = aug[row][j] - sub;
            }
        }
    }

    let mut x = vec![C64::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum = sum - aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
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
    fn test_build_hankel_matrix() {
        let mut y = Mat::<f64>::zeros(1, 5);
        for i in 0..5 {
            y[(0, i)] = (i + 1) as f64;
        }

        let h = build_hankel_matrix(&y, 2).unwrap();
        // 3 rows (delays+1), 3 cols (5-2)
        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 3);
        // H = [[1,2,3],[2,3,4],[3,4,5]]
        assert_near(h[(0, 0)], 1.0, 1e-12);
        assert_near(h[(1, 0)], 2.0, 1e-12);
        assert_near(h[(2, 0)], 3.0, 1e-12);
        assert_near(h[(0, 2)], 3.0, 1e-12);
        assert_near(h[(2, 2)], 5.0, 1e-12);
    }

    #[test]
    fn test_hankel_dmd_scalar() {
        let y = make_oscillatory_scalar(200);
        let config = HankelConfig {
            delays: Some(10),
            dt: 0.1,
            ..Default::default()
        };
        let result = hankel_dmd(&y, &config).unwrap();

        assert_eq!(result.delays, 10);
        assert_eq!(result.n_obs, 1);
        assert!(result.rank > 0);
        assert!(result.rank <= 11); // max = delays + 1
        assert!(result.residual < 0.5);
    }

    #[test]
    fn test_hankel_dmd_eigenvalues_near_unit_circle() {
        let y = make_oscillatory_scalar(200);
        let config = HankelConfig {
            delays: Some(10),
            dt: 0.1,
            ..Default::default()
        };
        let result = hankel_dmd(&y, &config).unwrap();

        // For a pure oscillation, dominant eigenvalues should be near unit circle
        let mut max_mag = 0.0_f64;
        for ev in &result.eigenvalues {
            max_mag = max_mag.max(ev.norm());
        }
        assert!(max_mag > 0.9 && max_mag < 1.1);
    }

    #[test]
    fn test_hankel_dmd_2d() {
        let y = make_oscillatory_2d(100);
        let config = HankelConfig {
            delays: Some(5),
            dt: 0.1,
            ..Default::default()
        };
        let result = hankel_dmd(&y, &config).unwrap();

        assert_eq!(result.n_obs, 2);
        assert_eq!(result.hankel_dim.0, 12); // (5+1)*2
    }

    #[test]
    fn test_hankel_reconstruct() {
        let y = make_oscillatory_scalar(100);
        let config = HankelConfig {
            delays: Some(10),
            dt: 0.1,
            ..Default::default()
        };
        let result = hankel_dmd(&y, &config).unwrap();

        let recon = hankel_reconstruct(&result, 50).unwrap();
        assert_eq!(recon.nrows(), 1);
        assert_eq!(recon.ncols(), 50);
    }

    #[test]
    fn test_hankel_predict() {
        let y = make_oscillatory_scalar(100);
        let config = HankelConfig {
            delays: Some(10),
            dt: 0.1,
            ..Default::default()
        };
        let result = hankel_dmd(&y, &config).unwrap();

        let pred = hankel_predict(&result, 20).unwrap();
        assert_eq!(pred.nrows(), 1);
        assert_eq!(pred.ncols(), 20);
    }

    #[test]
    fn test_hankel_auto_delays() {
        let y = make_oscillatory_scalar(60);
        let config = HankelConfig::default();
        let result = hankel_dmd(&y, &config).unwrap();

        // Auto delays = n_time/3 = 20
        assert_eq!(result.delays, 20);
    }

    #[test]
    fn test_hankel_too_few_points() {
        let y = Mat::<f64>::zeros(1, 3);
        let config = HankelConfig {
            delays: Some(5),
            ..Default::default()
        };
        assert!(hankel_dmd(&y, &config).is_err());
    }
}
