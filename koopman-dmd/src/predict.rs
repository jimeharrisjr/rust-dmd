use faer::Mat;

use crate::dmd::solve_amplitudes_pub;
use crate::types::{DmdError, DmdResult, C64};

/// Predict future states using DMD mode evolution.
///
/// x(k) = Σᵢ φᵢ · bᵢ · λᵢᵏ
///
/// When lifting was applied, predictions are computed in lifted space
/// then projected back to original observables.
///
/// # Arguments
/// * `result` - DMD result.
/// * `n_ahead` - Number of future time steps to predict.
/// * `x0` - Optional initial condition (in original, un-lifted space).
///   If None, uses the stored first snapshot.
///
/// # Returns
/// Matrix (n_vars_original × n_ahead) of predicted states.
pub fn predict_modes(
    result: &DmdResult,
    n_ahead: usize,
    x0: Option<&[f64]>,
) -> Result<Mat<f64>, DmdError> {
    if n_ahead == 0 {
        return Err(DmdError::InvalidInput(
            "n_ahead must be positive".to_string(),
        ));
    }

    let n_vars_lifted = result.n_vars();
    let rank = result.rank;

    // Use stored amplitudes (already computed for lifted first snapshot)
    let amplitudes = match x0 {
        Some(x0_val) => {
            // For lifted DMD with custom x0, we'd need to re-lift x0.
            // For now, solve in lifted space directly if dimensions match,
            // otherwise use original amplitudes.
            if x0_val.len() == n_vars_lifted {
                let x0_work: Vec<f64> = if result.center {
                    let means = result.x_mean.as_ref().unwrap();
                    x0_val.iter().zip(means).map(|(x, m)| x - m).collect()
                } else {
                    x0_val.to_vec()
                };
                solve_amplitudes_pub(&result.modes, &x0_work, n_vars_lifted, rank)?
            } else {
                // x0 is in original space but modes are lifted — use stored amplitudes
                result.amplitudes.clone()
            }
        }
        None => result.amplitudes.clone(),
    };

    // Predict in lifted space
    let mut pred_lifted = Mat::<f64>::zeros(n_vars_lifted, n_ahead);

    for k in 0..n_ahead {
        for i in 0..n_vars_lifted {
            let mut val = C64::zero();
            for j in 0..rank {
                let lambda_k = result.eigenvalues[j].powf((k + 1) as f64);
                val += result.modes[i][j] * amplitudes[j] * lambda_k;
            }
            pred_lifted[(i, k)] = val.re;
        }
    }

    // Add back means if centered
    if result.center {
        if let Some(ref means) = result.x_mean {
            for k in 0..n_ahead {
                for i in 0..n_vars_lifted {
                    pred_lifted[(i, k)] += means[i];
                }
            }
        }
    }

    // Project back to original space if lifting was applied
    match &result.lifting_info {
        Some(info) => {
            let n_orig = info.n_vars_original;
            let mut predictions = Mat::<f64>::zeros(n_orig, n_ahead);
            for k in 0..n_ahead {
                for (orig_i, &lifted_i) in info.observables.iter().enumerate() {
                    predictions[(orig_i, k)] = pred_lifted[(lifted_i, k)];
                }
            }
            Ok(predictions)
        }
        None => Ok(pred_lifted),
    }
}

/// Predict future states using iterative matrix multiplication.
///
/// x(k+1) = A · x(k)
///
/// When lifting was applied, the A matrix operates in lifted space
/// and results are projected back to original observables.
///
/// # Arguments
/// * `result` - DMD result.
/// * `n_ahead` - Number of future time steps to predict.
/// * `x0` - Optional initial condition (in lifted space if lifting was used).
///   If None, uses the last snapshot.
///
/// # Returns
/// Matrix (n_vars_original × n_ahead) of predicted states.
pub fn predict_matrix(
    result: &DmdResult,
    n_ahead: usize,
    x0: Option<&[f64]>,
) -> Result<Mat<f64>, DmdError> {
    if n_ahead == 0 {
        return Err(DmdError::InvalidInput(
            "n_ahead must be positive".to_string(),
        ));
    }

    let n_vars_lifted = result.n_vars();

    let mut x_current: Vec<C64> = match x0 {
        Some(x0_val) => {
            if x0_val.len() != n_vars_lifted {
                return Err(DmdError::InvalidInput(format!(
                    "x0 has length {}, expected {} (lifted space dimension)",
                    x0_val.len(),
                    n_vars_lifted
                )));
            }
            if result.center {
                let means = result.x_mean.as_ref().unwrap();
                x0_val
                    .iter()
                    .zip(means)
                    .map(|(x, m)| C64::new(x - m, 0.0))
                    .collect()
            } else {
                x0_val.iter().map(|&x| C64::new(x, 0.0)).collect()
            }
        }
        None => {
            // Use last column of the (lifted) working data
            // For non-lifted, x_last is the original last snapshot
            // For lifted, we need to reconstruct from modes at last time step
            if result.center {
                // Use modes to get last lifted state
                let n_time = result.data_dim.1;
                let mut x_last_lifted = vec![C64::zero(); n_vars_lifted];
                for i in 0..n_vars_lifted {
                    for j in 0..result.rank {
                        let lambda_k = result.eigenvalues[j].powf((n_time - 1) as f64);
                        x_last_lifted[i] += result.modes[i][j] * result.amplitudes[j] * lambda_k;
                    }
                }
                x_last_lifted
            } else if result.lifting_info.is_none() {
                result.x_last.iter().map(|&x| C64::new(x, 0.0)).collect()
            } else {
                // Lifted: reconstruct last state from modes
                let n_time = result.data_dim.1;
                let mut x_last_lifted = vec![C64::zero(); n_vars_lifted];
                for i in 0..n_vars_lifted {
                    for j in 0..result.rank {
                        let lambda_k = result.eigenvalues[j].powf((n_time - 1) as f64);
                        x_last_lifted[i] += result.modes[i][j] * result.amplitudes[j] * lambda_k;
                    }
                }
                x_last_lifted
            }
        }
    };

    let mut pred_lifted = Mat::<f64>::zeros(n_vars_lifted, n_ahead);

    for k in 0..n_ahead {
        let mut x_next = vec![C64::zero(); n_vars_lifted];
        for i in 0..n_vars_lifted {
            for j in 0..n_vars_lifted {
                x_next[i] += result.a_matrix[i][j] * x_current[j];
            }
        }
        x_current = x_next;

        for i in 0..n_vars_lifted {
            let mut val = x_current[i].re;
            if result.center {
                if let Some(ref means) = result.x_mean {
                    val += means[i];
                }
            }
            pred_lifted[(i, k)] = val;
        }
    }

    // Project back to original space if lifting was applied
    match &result.lifting_info {
        Some(info) => {
            let n_orig = info.n_vars_original;
            let mut predictions = Mat::<f64>::zeros(n_orig, n_ahead);
            for k in 0..n_ahead {
                for (orig_i, &lifted_i) in info.observables.iter().enumerate() {
                    predictions[(orig_i, k)] = pred_lifted[(lifted_i, k)];
                }
            }
            Ok(predictions)
        }
        None => Ok(pred_lifted),
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

    #[test]
    fn test_predict_modes_reconstruct() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        let pred = predict_modes(&result, 99, None).unwrap();
        assert_eq!(pred.nrows(), 2);
        assert_eq!(pred.ncols(), 99);

        // First predicted step should be close to x[:,1]
        assert_near(pred[(0, 0)], x[(0, 1)], 0.1);
        assert_near(pred[(1, 0)], x[(1, 1)], 0.1);
    }

    #[test]
    fn test_predict_matrix() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        let x0: Vec<f64> = (0..2).map(|i| x[(i, 0)]).collect();
        let pred = predict_matrix(&result, 10, Some(&x0)).unwrap();
        assert_eq!(pred.nrows(), 2);
        assert_eq!(pred.ncols(), 10);
    }

    #[test]
    fn test_predict_modes_and_matrix_agree() {
        let x = make_oscillatory_data(100);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();

        let x0: Vec<f64> = (0..2).map(|i| x[(i, 0)]).collect();
        let pred_modes = predict_modes(&result, 10, Some(&x0)).unwrap();
        let pred_matrix = predict_matrix(&result, 10, Some(&x0)).unwrap();

        for i in 0..2 {
            for k in 0..10 {
                assert_near(pred_modes[(i, k)], pred_matrix[(i, k)], 0.5);
            }
        }
    }

    #[test]
    fn test_predict_zero_ahead() {
        let x = make_oscillatory_data(50);
        let config = DmdConfig::default();
        let result = dmd(&x, &config).unwrap();
        assert!(predict_modes(&result, 0, None).is_err());
    }
}
