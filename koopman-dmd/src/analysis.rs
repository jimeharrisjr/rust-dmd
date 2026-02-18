use faer::Mat;

use crate::dmd::dmd;
use crate::types::{
    ConvergenceResult, DmdConfig, DmdError, DmdResult, DominantCriterion, ErrorMetrics, ModeInfo,
    PseudospectrumResult, ResidualResult, Stability, StabilityResult, C64,
};

/// Analyze the DMD eigenvalue spectrum.
///
/// Returns per-mode information: magnitude, phase, frequency, growth rate,
/// half-life, and stability classification.
pub fn dmd_spectrum(result: &DmdResult, dt: f64) -> Vec<ModeInfo> {
    let rank = result.rank;
    let mut info = Vec::with_capacity(rank);

    for i in 0..rank {
        let lambda = result.eigenvalues[i];
        let mag = lambda.norm();
        let phase = lambda.arg();
        let frequency = phase.abs() / (2.0 * std::f64::consts::PI * dt);
        let period = if frequency.abs() > 1e-14 {
            1.0 / frequency
        } else {
            f64::INFINITY
        };
        let growth_rate = mag.ln() / dt;
        let half_life = if growth_rate.abs() > 1e-14 {
            Some(-(2.0_f64.ln()) / growth_rate)
        } else {
            None
        };

        let stability = classify_eigenvalue(mag, 1e-6);
        let amplitude = result.amplitudes[i].norm();

        info.push(ModeInfo {
            index: i,
            eigenvalue: lambda,
            magnitude: mag,
            phase,
            frequency,
            period,
            growth_rate,
            half_life,
            stability,
            amplitude,
        });
    }

    info
}

/// Analyze system stability.
pub fn dmd_stability(result: &DmdResult, tol: f64) -> StabilityResult {
    let mode_stability: Vec<Stability> = result
        .eigenvalues
        .iter()
        .map(|lambda| classify_eigenvalue(lambda.norm(), tol))
        .collect();

    let spectral_radius = result
        .eigenvalues
        .iter()
        .map(|lambda| lambda.norm())
        .fold(0.0_f64, f64::max);

    let is_unstable = mode_stability.contains(&Stability::Growing);
    let is_marginal = mode_stability.contains(&Stability::Neutral);
    let is_stable = !is_unstable;

    StabilityResult {
        is_stable,
        is_unstable,
        is_marginal,
        spectral_radius,
        mode_stability,
    }
}

/// Reconstruct the data from DMD modes.
///
/// X_recon[:, k] = Σᵢ φᵢ · bᵢ · λᵢᵏ
///
/// If `modes_subset` is provided, only the specified mode indices are used.
pub fn dmd_reconstruct(
    result: &DmdResult,
    n_steps: usize,
    modes_subset: Option<&[usize]>,
) -> Result<Mat<f64>, DmdError> {
    let n_vars = result.n_vars();
    let rank = result.rank;

    let indices: Vec<usize> = match modes_subset {
        Some(idx) => {
            for &i in idx {
                if i >= rank {
                    return Err(DmdError::InvalidInput(format!(
                        "mode index {i} out of range (rank={rank})"
                    )));
                }
            }
            idx.to_vec()
        }
        None => (0..rank).collect(),
    };

    let mut recon = Mat::<f64>::zeros(n_vars, n_steps);

    for k in 0..n_steps {
        for i in 0..n_vars {
            let mut val = C64::zero();
            for &j in &indices {
                let lambda_k = result.eigenvalues[j].powf(k as f64);
                val += result.modes[i][j] * result.amplitudes[j] * lambda_k;
            }
            recon[(i, k)] = val.re;
        }
    }

    // Add back means if centered
    if result.center {
        if let Some(ref means) = result.x_mean {
            for k in 0..n_steps {
                for i in 0..n_vars {
                    recon[(i, k)] += means[i];
                }
            }
        }
    }

    Ok(recon)
}

/// Compute reconstruction error metrics.
pub fn dmd_error(result: &DmdResult, x_original: &Mat<f64>) -> Result<ErrorMetrics, DmdError> {
    let n_vars = x_original.nrows();
    let n_time = x_original.ncols();
    let recon = dmd_reconstruct(result, n_time, None)?;

    let mut sum_sq = 0.0;
    let mut sum_abs = 0.0;
    let mut sum_pct = 0.0;
    let mut orig_norm_sq = 0.0;
    let mut per_var_sq = vec![0.0; n_vars];
    let n_total = (n_vars * n_time) as f64;

    for i in 0..n_vars {
        for k in 0..n_time {
            let diff = recon[(i, k)] - x_original[(i, k)];
            sum_sq += diff * diff;
            sum_abs += diff.abs();
            orig_norm_sq += x_original[(i, k)] * x_original[(i, k)];
            per_var_sq[i] += diff * diff;

            let denom = x_original[(i, k)].abs();
            if denom > 1e-14 {
                sum_pct += (diff / denom).abs();
            }
        }
    }

    let rmse = (sum_sq / n_total).sqrt();
    let mae = sum_abs / n_total;
    let mape = sum_pct / n_total * 100.0;
    let relative_error = if orig_norm_sq > 0.0 {
        (sum_sq / orig_norm_sq).sqrt()
    } else {
        0.0
    };
    let per_variable_rmse: Vec<f64> = per_var_sq
        .iter()
        .map(|v| (v / n_time as f64).sqrt())
        .collect();

    Ok(ErrorMetrics {
        rmse,
        mae,
        mape,
        relative_error,
        per_variable_rmse,
    })
}

/// Extract the indices of the dominant modes.
pub fn dmd_dominant_modes(
    result: &DmdResult,
    n: usize,
    criterion: DominantCriterion,
) -> Vec<usize> {
    let rank = result.rank;
    let n = n.min(rank);

    let mut scored: Vec<(usize, f64)> = (0..rank)
        .map(|i| {
            let score = match criterion {
                DominantCriterion::Amplitude => result.amplitudes[i].norm(),
                DominantCriterion::Energy => {
                    result.amplitudes[i].norm() * result.eigenvalues[i].norm()
                }
                DominantCriterion::Stability => {
                    1.0 / (1.0 + (result.eigenvalues[i].norm() - 1.0).abs())
                }
            };
            (i, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.iter().take(n).map(|(i, _)| *i).collect()
}

/// Compute residuals of the DMD one-step predictions.
///
/// Computes x_{k+1} = A x_k and compares to actual x_{k+1}.
///
/// # Arguments
/// * `result` - DMD result.
/// * `x_original` - Original data matrix (n_vars × n_time).
pub fn dmd_residual(result: &DmdResult, x_original: &Mat<f64>) -> Result<ResidualResult, DmdError> {
    let n_vars = result.n_vars();
    let n_time = x_original.ncols();

    if x_original.nrows() != n_vars {
        return Err(DmdError::InvalidInput(format!(
            "x_original has {} rows, expected {}",
            x_original.nrows(),
            n_vars
        )));
    }
    if n_time < 2 {
        return Err(DmdError::InvalidInput("need at least 2 time points".into()));
    }

    // Center if needed
    let x_work = if result.center {
        let means = result.x_mean.as_ref().unwrap();
        let mut centered = x_original.clone();
        for j in 0..n_time {
            for i in 0..n_vars {
                centered[(i, j)] -= means[i];
            }
        }
        centered
    } else {
        x_original.clone()
    };

    // X1 = x_work[:, 0..n-1], X2 = x_work[:, 1..n]
    let n_pairs = n_time - 1;

    // Compute one-step predictions: A * X1
    let mut residual_norm_sq = 0.0;
    let mut data_norm_sq = 0.0;
    let mut per_step_residual = vec![0.0; n_pairs];

    for k in 0..n_pairs {
        let mut step_res_sq = 0.0;
        for i in 0..n_vars {
            // Compute (A * x_k)[i]
            let mut pred = 0.0;
            for j in 0..n_vars {
                pred += result.a_matrix[i][j].re * x_work[(j, k)];
            }
            let actual = x_work[(i, k + 1)];
            let diff = pred - actual;
            step_res_sq += diff * diff;
            data_norm_sq += actual * actual;
        }
        residual_norm_sq += step_res_sq;
        per_step_residual[k] = step_res_sq.sqrt();
    }

    let residual_norm = residual_norm_sq.sqrt();
    let residual_relative = if data_norm_sq > 0.0 {
        residual_norm / data_norm_sq.sqrt()
    } else {
        0.0
    };

    // Per-mode residual contribution: project residual onto each mode direction
    let mut per_mode_residual = vec![0.0; result.rank];
    for mode_idx in 0..result.rank {
        let mut mode_norm_sq = 0.0;
        for i in 0..n_vars {
            mode_norm_sq += result.modes[i][mode_idx].norm_sqr();
        }
        let mode_norm = mode_norm_sq.sqrt();
        if mode_norm < 1e-14 {
            continue;
        }

        // Project per-step residuals onto this mode
        let mut proj_sq = 0.0;
        for k in 0..n_pairs {
            let mut proj = C64::zero();
            for i in 0..n_vars {
                let mut pred = 0.0;
                for j in 0..n_vars {
                    pred += result.a_matrix[i][j].re * x_work[(j, k)];
                }
                let diff = pred - x_work[(i, k + 1)];
                proj += result.modes[i][mode_idx].conj() * C64::new(diff, 0.0);
            }
            proj_sq += proj.norm_sqr();
        }
        per_mode_residual[mode_idx] = proj_sq.sqrt() / mode_norm;
    }

    Ok(ResidualResult {
        residual_norm,
        residual_relative,
        per_step_residual,
        per_mode_residual,
    })
}

/// Compute the ε-pseudospectrum of the reduced DMD operator.
///
/// The ε-pseudospectrum is {z ∈ ℂ : σ_min(zI - Ã) ≤ ε}.
///
/// # Arguments
/// * `result` - DMD result.
/// * `epsilon` - Contour levels.
/// * `grid_n` - Number of grid points per axis.
/// * `margin` - Margin around eigenvalues (fraction of range).
pub fn dmd_pseudospectrum(
    result: &DmdResult,
    epsilon: &[f64],
    grid_n: usize,
    margin: f64,
) -> PseudospectrumResult {
    let rank = result.rank;

    // Determine grid limits from eigenvalues
    let re_vals: Vec<f64> = result.eigenvalues.iter().map(|e| e.re).collect();
    let im_vals: Vec<f64> = result.eigenvalues.iter().map(|e| e.im).collect();

    let re_min = re_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let re_max = re_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let im_min = im_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let im_max = im_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let re_margin = (re_max - re_min).max(1.0) * margin;
    let im_margin = (im_max - im_min).max(1.0) * margin;

    let x_min = re_min - re_margin;
    let x_max = re_max + re_margin;
    let y_min = im_min - im_margin;
    let y_max = im_max + im_margin;

    let x: Vec<f64> = (0..grid_n)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (grid_n - 1) as f64)
        .collect();
    let y: Vec<f64> = (0..grid_n)
        .map(|i| y_min + (y_max - y_min) * i as f64 / (grid_n - 1) as f64)
        .collect();

    // Build Ã as a dense complex matrix (rank × rank)
    // Compute σ_min(zI - Ã) at each grid point
    let mut sigma_min = vec![vec![0.0; grid_n]; grid_n];

    for gi in 0..grid_n {
        for gj in 0..grid_n {
            let z = C64::new(x[gi], y[gj]);

            // Build (zI - Ã) as a real matrix of size 2r × 2r
            // [Re(zI-Ã) -Im(zI-Ã)]
            // [Im(zI-Ã)  Re(zI-Ã)]
            let mut m = Mat::<f64>::zeros(2 * rank, 2 * rank);
            for i in 0..rank {
                for j in 0..rank {
                    let a_ij = result.a_tilde[i][j];
                    let val = if i == j { z - a_ij } else { C64::zero() - a_ij };
                    m[(i, j)] = val.re;
                    m[(i, j + rank)] = -val.im;
                    m[(i + rank, j)] = val.im;
                    m[(i + rank, j + rank)] = val.re;
                }
            }

            // SVD to get minimum singular value
            if let Ok(svd) = m.svd() {
                let s_col = svd.S().column_vector();
                let n_sv = s_col.nrows();
                let mut min_sv = f64::INFINITY;
                for k in 0..n_sv {
                    let sv = s_col[k];
                    if sv < min_sv {
                        min_sv = sv;
                    }
                }
                sigma_min[gi][gj] = min_sv;
            } else {
                sigma_min[gi][gj] = 0.0;
            }
        }
    }

    PseudospectrumResult {
        x,
        y,
        sigma_min,
        eigenvalues: result.eigenvalues.clone(),
        epsilon: epsilon.to_vec(),
    }
}

/// Estimate the convergence rate of the DMD approximation.
///
/// Computes DMD at increasing sample sizes and measures eigenvalue convergence.
/// For systems with pure point spectrum, convergence should be O(1/m^α).
///
/// # Arguments
/// * `x` - Full data matrix (n_vars × n_time).
/// * `sample_fractions` - Fractions of data to use.
/// * `config` - DMD configuration.
pub fn dmd_convergence(
    x: &Mat<f64>,
    sample_fractions: &[f64],
    config: &DmdConfig,
) -> Result<ConvergenceResult, DmdError> {
    let n_time = x.ncols();
    if n_time < 10 {
        return Err(DmdError::InvalidInput(
            "need at least 10 time points for convergence analysis".into(),
        ));
    }

    let mut sample_sizes: Vec<usize> = sample_fractions
        .iter()
        .map(|&f| ((f * n_time as f64).floor() as usize).max(5))
        .collect();
    sample_sizes.dedup();

    let mut eigenvalues_list = Vec::new();

    for &m in &sample_sizes {
        let m = m.min(n_time);
        let x_sub = x.subcols(0, m).to_owned();
        match dmd(&x_sub, config) {
            Ok(result) => eigenvalues_list.push(result.eigenvalues),
            Err(_) => eigenvalues_list.push(Vec::new()),
        }
    }

    // Compute eigenvalue magnitude changes between successive fits
    let mut eigenvalue_changes = Vec::new();
    for i in 1..sample_sizes.len() {
        let prev = &eigenvalues_list[i - 1];
        let curr = &eigenvalues_list[i];
        let n_compare = prev.len().min(curr.len());

        if n_compare > 0 {
            let mut prev_mags: Vec<f64> = prev.iter().map(|e| e.norm()).collect();
            let mut curr_mags: Vec<f64> = curr.iter().map(|e| e.norm()).collect();
            prev_mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
            curr_mags.sort_by(|a, b| b.partial_cmp(a).unwrap());

            let max_change = prev_mags[..n_compare]
                .iter()
                .zip(&curr_mags[..n_compare])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            eigenvalue_changes.push(max_change);
        } else {
            eigenvalue_changes.push(f64::NAN);
        }
    }

    // Estimate convergence rate via log-log regression
    // log(change) ~ -alpha * log(m)
    let convergence_estimate = if sample_sizes.len() >= 3 {
        let valid: Vec<(f64, f64)> = eigenvalue_changes
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0.0 && c.is_finite())
            .map(|(i, &c)| (sample_sizes[i + 1] as f64, c))
            .collect();

        if valid.len() >= 2 {
            // Simple linear regression: log_change = a + b * log_m
            let n = valid.len() as f64;
            let sum_lm: f64 = valid.iter().map(|(m, _)| m.ln()).sum();
            let sum_lc: f64 = valid.iter().map(|(_, c)| c.ln()).sum();
            let sum_lm2: f64 = valid.iter().map(|(m, _)| m.ln().powi(2)).sum();
            let sum_lm_lc: f64 = valid.iter().map(|(m, c)| m.ln() * c.ln()).sum();

            let denom = n * sum_lm2 - sum_lm * sum_lm;
            if denom.abs() > 1e-14 {
                let slope = (n * sum_lm_lc - sum_lm * sum_lc) / denom;
                Some(-slope) // convergence rate is negative of slope
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(ConvergenceResult {
        sample_sizes,
        eigenvalues: eigenvalues_list,
        eigenvalue_changes,
        convergence_estimate,
    })
}

/// Classify an eigenvalue by its magnitude relative to the unit circle.
fn classify_eigenvalue(magnitude: f64, tol: f64) -> Stability {
    if magnitude < 1.0 - tol {
        Stability::Decaying
    } else if magnitude > 1.0 + tol {
        Stability::Growing
    } else {
        Stability::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmd::dmd;
    use crate::types::DmdConfig;
    use std::f64::consts::PI;

    fn assert_near(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

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

    fn make_decaying_data(n_time: usize) -> Mat<f64> {
        let dt = 0.1;
        let decay = 0.05;
        let mut x = Mat::<f64>::zeros(2, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            let envelope = (-decay * time).exp();
            x[(0, t)] = envelope * (2.0 * PI * 0.5 * time).cos();
            x[(1, t)] = envelope * (2.0 * PI * 0.5 * time).sin();
        }
        x
    }

    #[test]
    fn test_spectrum() {
        let x = make_oscillatory_data(200);
        let config = DmdConfig {
            dt: 0.1,
            ..Default::default()
        };
        let result = dmd(&x, &config).unwrap();
        let spec = dmd_spectrum(&result, 0.1);

        assert_eq!(spec.len(), result.rank);
        for mode in &spec {
            assert_near(mode.magnitude, 1.0, 0.05);
            assert_near(mode.frequency, 0.5, 0.1);
        }
    }

    #[test]
    fn test_stability_stable() {
        let x = make_decaying_data(200);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();
        let stab = dmd_stability(&result, 0.01);

        assert!(stab.is_stable);
        assert!(!stab.is_unstable);
        assert!(stab.spectral_radius < 1.01);
    }

    #[test]
    fn test_stability_neutral() {
        let x = make_oscillatory_data(200);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();
        let stab = dmd_stability(&result, 0.05);

        assert!(stab.is_stable);
        assert_near(stab.spectral_radius, 1.0, 0.05);
    }

    #[test]
    fn test_reconstruct() {
        let x = make_oscillatory_data(50);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();
        let recon = dmd_reconstruct(&result, 50, None).unwrap();

        assert_eq!(recon.nrows(), 2);
        assert_eq!(recon.ncols(), 50);
        for i in 0..2 {
            for k in 0..50 {
                assert_near(recon[(i, k)], x[(i, k)], 0.2);
            }
        }
    }

    #[test]
    fn test_error_metrics() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();
        let err = dmd_error(&result, &x).unwrap();

        assert!(err.relative_error < 0.2);
        assert!(err.rmse < 0.5);
        assert_eq!(err.per_variable_rmse.len(), 2);
    }

    #[test]
    fn test_dominant_modes() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        let dominant = dmd_dominant_modes(&result, 1, DominantCriterion::Amplitude);
        assert_eq!(dominant.len(), 1);
        assert!(dominant[0] < result.rank);
    }

    #[test]
    fn test_reconstruct_subset_modes() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig {
            rank: Some(2),
            ..Default::default()
        };
        let result = dmd(&x, &config).unwrap();

        let recon = dmd_reconstruct(&result, 50, Some(&[0])).unwrap();
        assert_eq!(recon.nrows(), 2);
        assert_eq!(recon.ncols(), 50);
    }

    #[test]
    fn test_residual() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();
        let res = dmd_residual(&result, &x).unwrap();

        assert!(res.residual_relative < 0.5);
        assert_eq!(res.per_step_residual.len(), 99);
        assert_eq!(res.per_mode_residual.len(), result.rank);
    }

    #[test]
    fn test_pseudospectrum() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        let ps = dmd_pseudospectrum(&result, &[0.01, 0.1], 20, 0.3);
        assert_eq!(ps.x.len(), 20);
        assert_eq!(ps.y.len(), 20);
        assert_eq!(ps.sigma_min.len(), 20);
        assert_eq!(ps.sigma_min[0].len(), 20);

        // σ_min should be very small near eigenvalues
        // and larger away from them
        let max_sigma: f64 = ps
            .sigma_min
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(max_sigma > 0.0);
    }

    #[test]
    fn test_convergence() {
        let x = make_oscillatory_data(200);
        let config = DmdConfig::default();
        let fractions = [0.25, 0.5, 0.75, 1.0];
        let conv = dmd_convergence(&x, &fractions, &config).unwrap();

        assert_eq!(conv.sample_sizes.len(), 4);
        assert_eq!(conv.eigenvalue_changes.len(), 3);
        assert_eq!(conv.eigenvalues.len(), 4);
    }
}
