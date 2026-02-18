use faer::Mat;

use crate::lifting::lift_data;
use crate::types::{DmdConfig, DmdError, DmdResult, SvdComponents, C64};
use crate::utils::{determine_rank, row_means, validate_matrix};

/// Perform Dynamic Mode Decomposition on a data matrix.
///
/// # Arguments
/// * `x` - Data matrix (m variables × n snapshots), columns are time-ordered.
/// * `config` - DMD configuration (rank, centering, dt).
///
/// # Algorithm
/// 1. Split X into X₁ = X[:, 0..n-1] and X₂ = X[:, 1..n]
/// 2. Truncated SVD: X₁ ≈ U Σ Vᵀ
/// 3. Reduced operator: Ã = Uᵀ X₂ V Σ⁻¹
/// 4. Eigendecomposition: Ã W = W Λ
/// 5. DMD modes: Φ = X₂ V Σ⁻¹ W
/// 6. Amplitudes: b = Φ⁺ x₀
pub fn dmd(x: &Mat<f64>, config: &DmdConfig) -> Result<DmdResult, DmdError> {
    validate_matrix(x, 1, 3)?;

    // Apply lifting if configured
    let (x_data, lifting_info) = match &config.lifting {
        Some(lifting_config) => {
            let (lifted, info) = lift_data(x, lifting_config)?;
            if lifted.ncols() < 3 {
                return Err(DmdError::InvalidInput(
                    "lifting reduced data to fewer than 3 columns".into(),
                ));
            }
            (lifted, Some(info))
        }
        None => (x.clone(), None),
    };

    let n_vars = x_data.nrows();
    let n_time = x_data.ncols();

    // Store original (pre-lifting) first/last snapshots for reference
    let x_first: Vec<f64> = (0..x.nrows()).map(|i| x[(i, 0)]).collect();
    let x_last: Vec<f64> = (0..x.nrows()).map(|i| x[(i, x.ncols() - 1)]).collect();

    // Optional centering (applied to potentially-lifted data)
    let (x_work, x_mean) = if config.center {
        let means = row_means(&x_data);
        let mut centered = x_data.clone();
        for j in 0..n_time {
            for i in 0..n_vars {
                centered[(i, j)] -= means[i];
            }
        }
        (centered, Some(means))
    } else {
        (x_data, None)
    };

    // Split into X1 (current) and X2 (next)
    let x1 = x_work.subcols(0, n_time - 1).to_owned();
    let x2 = x_work.subcols(1, n_time - 1).to_owned();

    // SVD of X1
    let svd = x1
        .svd()
        .map_err(|e| DmdError::SvdFailed(format!("{e:?}")))?;
    let u_full = svd.U();
    let v_full = svd.V();
    let s_col = svd.S().column_vector();

    let n_sv = s_col.nrows();
    let s_vals: Vec<f64> = (0..n_sv).map(|i| s_col[i]).collect();

    let rank = determine_rank(&s_vals, config.rank, 0.99);

    // Truncate to rank r
    let u = u_full.subcols(0, rank).to_owned();
    let v = v_full.subcols(0, rank).to_owned();
    let s: Vec<f64> = s_vals[..rank].to_vec();

    // Reduced DMD matrix: Ã = Uᵀ X₂ V Σ⁻¹
    let ut_x2 = u.transpose() * &x2; // (r × n-1)
    let ut_x2_v = &ut_x2 * &v; // (r × r)

    // Multiply by Σ⁻¹ (scale columns)
    let mut a_tilde_real = Mat::<f64>::zeros(rank, rank);
    for i in 0..rank {
        for j in 0..rank {
            a_tilde_real[(i, j)] = ut_x2_v[(i, j)] / s[j];
        }
    }

    // Eigendecomposition of Ã
    let eigen = a_tilde_real
        .as_ref()
        .eigen()
        .map_err(|e| DmdError::EigenFailed(format!("{e:?}")))?;

    let eigenvalues_diag = eigen.S().column_vector();
    let eigenvectors = eigen.U();

    // Extract eigenvalues and eigenvectors into our types
    let mut eigenvalues = Vec::with_capacity(rank);
    let mut w_re = Mat::<f64>::zeros(rank, rank);
    let mut w_im = Mat::<f64>::zeros(rank, rank);

    for j in 0..rank {
        let ev = eigenvalues_diag[j];
        eigenvalues.push(C64::new(ev.re, ev.im));
        for i in 0..rank {
            let v = eigenvectors[(i, j)];
            w_re[(i, j)] = v.re;
            w_im[(i, j)] = v.im;
        }
    }

    // DMD modes: Φ = X₂ V Σ⁻¹ W
    // First compute X₂ V Σ⁻¹
    let x2_v = &x2 * &v; // (m × r)
    let mut x2_v_sinv = Mat::<f64>::zeros(n_vars, rank);
    for i in 0..n_vars {
        for j in 0..rank {
            x2_v_sinv[(i, j)] = x2_v[(i, j)] / s[j];
        }
    }

    // Multiply by W (complex): modes_re = x2_v_sinv * w_re, modes_im = x2_v_sinv * w_im
    let modes_re = &x2_v_sinv * &w_re;
    let modes_im = &x2_v_sinv * &w_im;

    // Pack into Vec<Vec<C64>> (row-major: modes[i][j] = mode j at variable i)
    let mut modes = vec![vec![C64::zero(); rank]; n_vars];
    for i in 0..n_vars {
        for j in 0..rank {
            modes[i][j] = C64::new(modes_re[(i, j)], modes_im[(i, j)]);
        }
    }

    // Amplitudes: b = Φ⁺ x₀ (use the lifted first snapshot)
    let lifted_first: Vec<f64> = (0..n_vars).map(|i| x_work[(i, 0)]).collect();
    let x0: Vec<f64> = if config.center {
        // x_work is already centered, so the first column of x_work is already centered
        lifted_first
    } else {
        lifted_first
    };

    let amplitudes = solve_amplitudes(&modes, &x0, n_vars, rank)?;

    // Reduced A_tilde as complex
    let mut a_tilde = vec![vec![C64::zero(); rank]; rank];
    for i in 0..rank {
        for j in 0..rank {
            a_tilde[i][j] = C64::new(a_tilde_real[(i, j)], 0.0);
        }
    }

    // Full A matrix: A = Φ Λ Φ⁺
    let a_matrix = compute_full_a(&modes, &eigenvalues, n_vars, rank)?;

    let svd_components = SvdComponents { u, s, v };

    Ok(DmdResult {
        a_matrix,
        modes,
        eigenvalues,
        amplitudes,
        rank,
        svd: svd_components,
        a_tilde,
        x_first,
        x_last,
        data_dim: (n_vars, n_time),
        center: config.center,
        x_mean,
        dt: config.dt,
        lifting_info,
    })
}

/// Public wrapper for amplitude solving, used by predict module.
pub fn solve_amplitudes_pub(
    modes: &[Vec<C64>],
    x0: &[f64],
    n_vars: usize,
    rank: usize,
) -> Result<Vec<C64>, DmdError> {
    solve_amplitudes(modes, x0, n_vars, rank)
}

/// Solve for amplitudes b via least-squares: Φ b ≈ x₀.
fn solve_amplitudes(
    modes: &[Vec<C64>],
    x0: &[f64],
    n_vars: usize,
    rank: usize,
) -> Result<Vec<C64>, DmdError> {
    // Build Gram matrix: Φ^H Φ (r × r, complex)
    // and RHS: Φ^H x0 (r, complex)
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

    // Solve via Gaussian elimination with partial pivoting
    complex_solve(&gram, &rhs)
}

/// Solve a complex linear system Ax = b using Gaussian elimination.
fn complex_solve(a: &[Vec<C64>], b: &[C64]) -> Result<Vec<C64>, DmdError> {
    let n = b.len();
    // Augmented matrix
    let mut aug: Vec<Vec<C64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.push(b[i]);
            new_row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
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
            return Err(DmdError::SolveFailed("singular matrix".into()));
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

    // Back substitution
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

/// Compute the full A matrix: A = Φ Λ Φ⁺.
fn compute_full_a(
    modes: &[Vec<C64>],
    eigenvalues: &[C64],
    n_vars: usize,
    rank: usize,
) -> Result<Vec<Vec<C64>>, DmdError> {
    // Φ Λ: scale each column of Φ by its eigenvalue
    let mut phi_lambda = vec![vec![C64::zero(); rank]; n_vars];
    for i in 0..n_vars {
        for j in 0..rank {
            phi_lambda[i][j] = modes[i][j] * eigenvalues[j];
        }
    }

    // Φ⁺ via (Φ^H Φ)⁻¹ Φ^H, computed column by column
    let mut gram = vec![vec![C64::zero(); rank]; rank];
    for i in 0..rank {
        for j in 0..rank {
            let mut val = C64::zero();
            for k in 0..n_vars {
                val += modes[k][i].conj() * modes[k][j];
            }
            gram[i][j] = val;
        }
    }

    // Compute Φ⁺ = (Φ^H Φ)⁻¹ Φ^H
    let mut phi_pinv = vec![vec![C64::zero(); n_vars]; rank];
    for col in 0..n_vars {
        // RHS = Φ^H[:, col]
        let rhs: Vec<C64> = (0..rank).map(|i| modes[col][i].conj()).collect();
        match complex_solve(&gram, &rhs) {
            Ok(x) => {
                for i in 0..rank {
                    phi_pinv[i][col] = x[i];
                }
            }
            Err(_) => {
                // Fallback: simple Φ^H / ||Φ||²
                let norm_sq: f64 = (0..n_vars).map(|k| modes[k][0].norm_sqr()).sum();
                for i in 0..rank {
                    phi_pinv[i][col] = modes[col][i].conj() / norm_sq;
                }
            }
        }
    }

    // A = phi_lambda * phi_pinv (n_vars × n_vars)
    let mut a = vec![vec![C64::zero(); n_vars]; n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut val = C64::zero();
            for k in 0..rank {
                val += phi_lambda[i][k] * phi_pinv[k][j];
            }
            a[i][j] = val;
        }
    }

    Ok(a)
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

    /// Create a simple oscillatory test system: two sinusoids.
    fn make_oscillatory_data(n_time: usize) -> Mat<f64> {
        let dt = 0.1;
        let mut x = Mat::<f64>::zeros(2, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            x[(0, t)] = (2.0 * PI * 0.5 * time).cos();
            x[(1, t)] = (2.0 * PI * 0.5 * time).sin();
        }
        x
    }

    #[test]
    fn test_dmd_basic() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        assert_eq!(result.data_dim, (2, 100));
        assert!(result.rank > 0);
        assert!(result.rank <= 2);
        assert_eq!(result.eigenvalues.len(), result.rank);
    }

    #[test]
    fn test_dmd_eigenvalue_magnitude() {
        let x = make_oscillatory_data(200);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        for ev in &result.eigenvalues {
            assert_near(ev.norm(), 1.0, 0.05);
        }
    }

    #[test]
    fn test_dmd_with_centering() {
        let mut x = make_oscillatory_data(100);
        for j in 0..100 {
            x[(0, j)] += 5.0;
            x[(1, j)] += 3.0;
        }

        let config = DmdConfig {
            center: true,
            ..Default::default()
        };
        let result = dmd(&x, &config).unwrap();
        assert!(result.center);
        assert!(result.x_mean.is_some());
    }

    #[test]
    fn test_dmd_explicit_rank() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig {
            rank: Some(2),
            ..Default::default()
        };
        let result = dmd(&x, &config).unwrap();
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_dmd_reconstruction_error() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        // Reconstruct first snapshot from modes and amplitudes
        let mut x0_recon = vec![C64::zero(); 2];
        for j in 0..result.rank {
            for i in 0..2 {
                x0_recon[i] += result.modes[i][j] * result.amplitudes[j];
            }
        }

        for i in 0..2 {
            assert_near(x0_recon[i].re, result.x_first[i], 0.1);
        }
    }

    #[test]
    fn test_dmd_too_few_columns() {
        let x = Mat::<f64>::zeros(3, 2);
        let config = DmdConfig::default();
        assert!(dmd(&x, &config).is_err());
    }
}
