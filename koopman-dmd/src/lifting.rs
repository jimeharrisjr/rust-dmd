use faer::Mat;

use crate::types::DmdError;

/// Configuration for lifting transformations.
#[derive(Debug, Clone)]
pub enum LiftingConfig {
    /// Polynomial lifting: [X, X², ..., Xᵈ]. `degree` must be >= 2.
    Polynomial { degree: usize },
    /// Polynomial lifting with all cross-terms up to total degree.
    PolynomialCross { degree: usize },
    /// Trigonometric lifting: [X, sin(X), cos(X), sin(2X), cos(2X), ...].
    Trigonometric { harmonics: usize },
    /// Time-delay embedding: [x(t), x(t-1), ..., x(t-d)].
    /// Note: reduces the number of columns by `delays`.
    Delay { delays: usize },
    /// Gaussian RBF features: [X, exp(-||x-c||²/2σ²)].
    /// `centers` is (n_vars × n_centers), `sigma` is the width.
    Rbf { centers: Mat<f64>, sigma: f64 },
}

/// Metadata about a lifting transformation that was applied.
#[derive(Debug, Clone)]
pub struct LiftingInfo {
    /// Description of the lifting used.
    pub description: String,
    /// Number of original state variables.
    pub n_vars_original: usize,
    /// Number of lifted state variables.
    pub n_vars_lifted: usize,
    /// Which rows of the lifted data correspond to original observables.
    /// Used to project predictions back to original space.
    pub observables: Vec<usize>,
}

/// Apply a lifting transformation to a data matrix.
///
/// # Arguments
/// * `x` - Data matrix (n_vars × n_time).
/// * `config` - Lifting configuration.
///
/// # Returns
/// Lifted data matrix (n_vars_lifted × n_time_out) and lifting info.
/// For most liftings, n_time_out == n_time. For delay embedding, n_time_out < n_time.
pub fn lift_data(
    x: &Mat<f64>,
    config: &LiftingConfig,
) -> Result<(Mat<f64>, LiftingInfo), DmdError> {
    match config {
        LiftingConfig::Polynomial { degree } => lift_polynomial(x, *degree),
        LiftingConfig::PolynomialCross { degree } => lift_poly_cross(x, *degree),
        LiftingConfig::Trigonometric { harmonics } => lift_trigonometric(x, *harmonics),
        LiftingConfig::Delay { delays } => lift_delay(x, *delays),
        LiftingConfig::Rbf { centers, sigma } => lift_rbf(x, centers, *sigma),
    }
}

/// Polynomial lifting: [X, X², X³, ..., Xᵈ].
fn lift_polynomial(x: &Mat<f64>, degree: usize) -> Result<(Mat<f64>, LiftingInfo), DmdError> {
    if degree < 2 {
        return Err(DmdError::InvalidInput(
            "polynomial degree must be at least 2".into(),
        ));
    }

    let n_vars = x.nrows();
    let n_time = x.ncols();
    let n_lifted = n_vars * degree;

    let mut lifted = Mat::<f64>::zeros(n_lifted, n_time);

    // Copy original data (degree 1)
    for i in 0..n_vars {
        for j in 0..n_time {
            lifted[(i, j)] = x[(i, j)];
        }
    }

    // Add higher-degree terms
    for d in 2..=degree {
        let row_offset = (d - 1) * n_vars;
        for i in 0..n_vars {
            for j in 0..n_time {
                lifted[(row_offset + i, j)] = x[(i, j)].powi(d as i32);
            }
        }
    }

    let observables = (0..n_vars).collect();
    let info = LiftingInfo {
        description: format!("polynomial(degree={degree})"),
        n_vars_original: n_vars,
        n_vars_lifted: n_lifted,
        observables,
    };

    Ok((lifted, info))
}

/// Generate all monomial exponent combinations up to a given total degree.
fn generate_monomials(n_vars: usize, max_degree: usize) -> Vec<Vec<usize>> {
    if n_vars == 1 {
        return (0..=max_degree).map(|d| vec![d]).collect();
    }

    let mut result = Vec::new();

    for total_deg in 0..=max_degree {
        generate_monomials_recursive(n_vars, total_deg, &mut vec![], &mut result);
    }

    result
}

fn generate_monomials_recursive(
    n_remaining: usize,
    remaining_degree: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if n_remaining == 1 {
        let mut mono = current.clone();
        mono.push(remaining_degree);
        result.push(mono);
        return;
    }

    for d in 0..=remaining_degree {
        current.push(d);
        generate_monomials_recursive(n_remaining - 1, remaining_degree - d, current, result);
        current.pop();
    }
}

/// Polynomial lifting with all cross-terms up to total degree.
fn lift_poly_cross(x: &Mat<f64>, degree: usize) -> Result<(Mat<f64>, LiftingInfo), DmdError> {
    if degree < 2 {
        return Err(DmdError::InvalidInput(
            "polynomial cross degree must be at least 2".into(),
        ));
    }

    let n_vars = x.nrows();
    let n_time = x.ncols();

    // Generate all monomial exponents with total degree >= 1
    let all_monomials = generate_monomials(n_vars, degree);
    let monomials: Vec<Vec<usize>> = all_monomials
        .into_iter()
        .filter(|m| m.iter().sum::<usize>() >= 1)
        .collect();

    let n_terms = monomials.len();
    let mut lifted = Mat::<f64>::zeros(n_terms, n_time);

    for (row, mono) in monomials.iter().enumerate() {
        for t in 0..n_time {
            let mut val = 1.0;
            for (var, &exp) in mono.iter().enumerate() {
                if exp > 0 {
                    val *= x[(var, t)].powi(exp as i32);
                }
            }
            lifted[(row, t)] = val;
        }
    }

    // Find which rows correspond to each original variable (degree-1 unit vectors)
    let mut observables = vec![0usize; n_vars];
    for (row, mono) in monomials.iter().enumerate() {
        let total: usize = mono.iter().sum();
        if total == 1 {
            // This is a single-variable degree-1 monomial
            for (var, &exp) in mono.iter().enumerate() {
                if exp == 1 {
                    observables[var] = row;
                }
            }
        }
    }

    let info = LiftingInfo {
        description: format!("polynomial_cross(degree={degree})"),
        n_vars_original: n_vars,
        n_vars_lifted: n_terms,
        observables,
    };

    Ok((lifted, info))
}

/// Trigonometric lifting: [X, sin(X), cos(X), sin(2X), cos(2X), ...].
fn lift_trigonometric(x: &Mat<f64>, harmonics: usize) -> Result<(Mat<f64>, LiftingInfo), DmdError> {
    if harmonics < 1 {
        return Err(DmdError::InvalidInput(
            "harmonics must be at least 1".into(),
        ));
    }

    let n_vars = x.nrows();
    let n_time = x.ncols();
    let n_lifted = n_vars * (1 + 2 * harmonics);

    let mut lifted = Mat::<f64>::zeros(n_lifted, n_time);

    // Copy original data
    for i in 0..n_vars {
        for j in 0..n_time {
            lifted[(i, j)] = x[(i, j)];
        }
    }

    // Add sin/cos harmonics
    for h in 1..=harmonics {
        let sin_offset = n_vars + (h - 1) * 2 * n_vars;
        let cos_offset = sin_offset + n_vars;
        let h_f = h as f64;
        for i in 0..n_vars {
            for j in 0..n_time {
                lifted[(sin_offset + i, j)] = (h_f * x[(i, j)]).sin();
                lifted[(cos_offset + i, j)] = (h_f * x[(i, j)]).cos();
            }
        }
    }

    let observables = (0..n_vars).collect();
    let info = LiftingInfo {
        description: format!("trigonometric(harmonics={harmonics})"),
        n_vars_original: n_vars,
        n_vars_lifted: n_lifted,
        observables,
    };

    Ok((lifted, info))
}

/// Time-delay embedding: [x(t), x(t-1), ..., x(t-d)].
///
/// Note: this reduces the number of time points by `delays`.
fn lift_delay(x: &Mat<f64>, delays: usize) -> Result<(Mat<f64>, LiftingInfo), DmdError> {
    if delays < 1 {
        return Err(DmdError::InvalidInput("delays must be at least 1".into()));
    }

    let n_vars = x.nrows();
    let n_time = x.ncols();

    if n_time <= delays {
        return Err(DmdError::InvalidInput(format!(
            "not enough time points ({n_time}) for {delays} delays"
        )));
    }

    let n_valid = n_time - delays;
    let n_lifted = n_vars * (delays + 1);

    let mut lifted = Mat::<f64>::zeros(n_lifted, n_valid);

    for d in 0..=delays {
        let row_start = d * n_vars;
        for i in 0..n_vars {
            for t in 0..n_valid {
                // d=0: most recent (cols delays..n_time-1)
                // d=1: one step back, etc.
                lifted[(row_start + i, t)] = x[(i, delays - d + t)];
            }
        }
    }

    let observables = (0..n_vars).collect();
    let info = LiftingInfo {
        description: format!("delay(delays={delays})"),
        n_vars_original: n_vars,
        n_vars_lifted: n_lifted,
        observables,
    };

    Ok((lifted, info))
}

/// Gaussian RBF lifting: [X, exp(-||x - c_i||² / (2σ²))].
fn lift_rbf(
    x: &Mat<f64>,
    centers: &Mat<f64>,
    sigma: f64,
) -> Result<(Mat<f64>, LiftingInfo), DmdError> {
    if sigma <= 0.0 {
        return Err(DmdError::InvalidInput("sigma must be positive".into()));
    }

    let n_vars = x.nrows();
    let n_time = x.ncols();

    if centers.nrows() != n_vars {
        return Err(DmdError::InvalidInput(format!(
            "centers must have {n_vars} rows (same as X), got {}",
            centers.nrows()
        )));
    }

    let n_centers = centers.ncols();
    let n_lifted = n_vars + n_centers;
    let two_sigma_sq = 2.0 * sigma * sigma;

    let mut lifted = Mat::<f64>::zeros(n_lifted, n_time);

    // Copy original data
    for i in 0..n_vars {
        for t in 0..n_time {
            lifted[(i, t)] = x[(i, t)];
        }
    }

    // Compute RBF activations
    for c in 0..n_centers {
        for t in 0..n_time {
            let mut sq_dist = 0.0;
            for i in 0..n_vars {
                let diff = x[(i, t)] - centers[(i, c)];
                sq_dist += diff * diff;
            }
            lifted[(n_vars + c, t)] = (-sq_dist / two_sigma_sq).exp();
        }
    }

    let observables = (0..n_vars).collect();
    let info = LiftingInfo {
        description: format!("rbf(n_centers={n_centers}, sigma={sigma})"),
        n_vars_original: n_vars,
        n_vars_lifted: n_lifted,
        observables,
    };

    Ok((lifted, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_near(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn make_test_data() -> Mat<f64> {
        let mut x = Mat::<f64>::zeros(2, 10);
        for t in 0..10 {
            x[(0, t)] = (t as f64) * 0.1;
            x[(1, t)] = (t as f64) * 0.2;
        }
        x
    }

    #[test]
    fn test_polynomial_lifting() {
        let x = make_test_data();
        let (lifted, info) = lift_polynomial(&x, 3).unwrap();

        assert_eq!(lifted.nrows(), 6); // 2 * 3
        assert_eq!(lifted.ncols(), 10);
        assert_eq!(info.n_vars_original, 2);
        assert_eq!(info.n_vars_lifted, 6);
        assert_eq!(info.observables, vec![0, 1]);

        // Check: row 0 = x, row 2 = x^2, row 4 = x^3
        let x_val = x[(0, 5)];
        assert_near(lifted[(0, 5)], x_val, 1e-12);
        assert_near(lifted[(2, 5)], x_val * x_val, 1e-12);
        assert_near(lifted[(4, 5)], x_val * x_val * x_val, 1e-12);
    }

    #[test]
    fn test_polynomial_degree_1_error() {
        let x = make_test_data();
        assert!(lift_polynomial(&x, 1).is_err());
    }

    #[test]
    fn test_poly_cross_lifting() {
        let x = make_test_data();
        let (lifted, info) = lift_poly_cross(&x, 2).unwrap();

        // For 2 variables, degree 2: x1, x2, x1^2, x1*x2, x2^2 = 5 terms
        assert_eq!(lifted.nrows(), 5);
        assert_eq!(lifted.ncols(), 10);
        assert_eq!(info.n_vars_original, 2);

        // Observable rows should contain original variables
        for t in 0..10 {
            assert_near(lifted[(info.observables[0], t)], x[(0, t)], 1e-12);
            assert_near(lifted[(info.observables[1], t)], x[(1, t)], 1e-12);
        }

        // Check a cross term exists: x1 * x2
        let t = 5;
        let x1 = x[(0, t)];
        let x2 = x[(1, t)];
        // One of the rows should contain x1 * x2
        let mut found_cross = false;
        for i in 2..lifted.nrows() {
            if (lifted[(i, t)] - x1 * x2).abs() < 1e-12 {
                found_cross = true;
                break;
            }
        }
        assert!(found_cross, "cross term x1*x2 not found");
    }

    #[test]
    fn test_trigonometric_lifting() {
        let x = make_test_data();
        let (lifted, info) = lift_trigonometric(&x, 2).unwrap();

        // 2 vars * (1 + 2*2) = 10 rows
        assert_eq!(lifted.nrows(), 10);
        assert_eq!(lifted.ncols(), 10);
        assert_eq!(info.n_vars_original, 2);
        assert_eq!(info.n_vars_lifted, 10);

        // Check sin/cos of first harmonic
        let t = 3;
        assert_near(lifted[(0, t)], x[(0, t)], 1e-12);
        assert_near(lifted[(2, t)], x[(0, t)].sin(), 1e-12); // sin(x1)
        assert_near(lifted[(4, t)], x[(0, t)].cos(), 1e-12); // cos(x1)
        assert_near(lifted[(3, t)], x[(1, t)].sin(), 1e-12); // sin(x2)
        assert_near(lifted[(5, t)], x[(1, t)].cos(), 1e-12); // cos(x2)

        // Second harmonic
        assert_near(lifted[(6, t)], (2.0 * x[(0, t)]).sin(), 1e-12);
        assert_near(lifted[(8, t)], (2.0 * x[(0, t)]).cos(), 1e-12);
    }

    #[test]
    fn test_delay_lifting() {
        let x = make_test_data();
        let (lifted, info) = lift_delay(&x, 3).unwrap();

        // 2 vars * (3+1) = 8 rows, 10-3 = 7 columns
        assert_eq!(lifted.nrows(), 8);
        assert_eq!(lifted.ncols(), 7);
        assert_eq!(info.n_vars_original, 2);
        assert_eq!(info.n_vars_lifted, 8);

        // Row 0-1: x(t), Row 2-3: x(t-1), Row 4-5: x(t-2), Row 6-7: x(t-3)
        // At output column 0, the "current" time is t=3 (delays offset)
        assert_near(lifted[(0, 0)], x[(0, 3)], 1e-12); // x1(t=3)
        assert_near(lifted[(2, 0)], x[(0, 2)], 1e-12); // x1(t=2)
        assert_near(lifted[(4, 0)], x[(0, 1)], 1e-12); // x1(t=1)
        assert_near(lifted[(6, 0)], x[(0, 0)], 1e-12); // x1(t=0)
    }

    #[test]
    fn test_delay_too_few_points() {
        let x = make_test_data();
        assert!(lift_delay(&x, 10).is_err());
    }

    #[test]
    fn test_rbf_lifting() {
        let x = make_test_data();

        // 3 centers, each 2D
        let mut centers = Mat::<f64>::zeros(2, 3);
        centers[(0, 0)] = 0.0;
        centers[(1, 0)] = 0.0;
        centers[(0, 1)] = 0.5;
        centers[(1, 1)] = 0.5;
        centers[(0, 2)] = 1.0;
        centers[(1, 2)] = 1.0;

        let (lifted, info) = lift_rbf(&x, &centers, 1.0).unwrap();

        assert_eq!(lifted.nrows(), 5); // 2 + 3
        assert_eq!(lifted.ncols(), 10);
        assert_eq!(info.n_vars_original, 2);

        // Original data preserved
        for t in 0..10 {
            assert_near(lifted[(0, t)], x[(0, t)], 1e-12);
            assert_near(lifted[(1, t)], x[(1, t)], 1e-12);
        }

        // RBF values should be in (0, 1]
        for c in 0..3 {
            for t in 0..10 {
                let val = lifted[(2 + c, t)];
                assert!(val > 0.0 && val <= 1.0 + 1e-12);
            }
        }

        // At center 0 (0,0), column 0 (x=[0,0]) should give RBF = 1.0
        assert_near(lifted[(2, 0)], 1.0, 1e-12);
    }

    #[test]
    fn test_rbf_sigma_error() {
        let x = make_test_data();
        let centers = Mat::<f64>::zeros(2, 1);
        assert!(lift_rbf(&x, &centers, 0.0).is_err());
        assert!(lift_rbf(&x, &centers, -1.0).is_err());
    }

    #[test]
    fn test_rbf_dimension_mismatch() {
        let x = make_test_data();
        let centers = Mat::<f64>::zeros(3, 2); // 3 rows != 2 vars
        assert!(lift_rbf(&x, &centers, 1.0).is_err());
    }

    #[test]
    fn test_lift_data_dispatch() {
        let x = make_test_data();
        let (lifted, info) = lift_data(&x, &LiftingConfig::Polynomial { degree: 2 }).unwrap();
        assert_eq!(lifted.nrows(), 4);
        assert_eq!(info.n_vars_original, 2);
    }

    #[test]
    fn test_generate_monomials() {
        // 2 vars, degree 2: should have (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
        let monos = generate_monomials(2, 2);
        assert_eq!(monos.len(), 6);

        // Verify all total degrees are <= 2
        for m in &monos {
            assert!(m.iter().sum::<usize>() <= 2);
        }
    }

    // ---- Integration tests: Lifted DMD ----

    #[test]
    fn test_dmd_with_polynomial_lifting() {
        use crate::dmd::dmd;
        use crate::predict::predict_modes;
        use crate::types::DmdConfig;

        // Create a system with mild nonlinearity: x(t) = cos(t) + 0.3*cos²(t)
        let n_time = 200;
        let dt = 0.1;
        let mut x = Mat::<f64>::zeros(2, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            x[(0, t)] = time.cos() + 0.3 * time.cos().powi(2);
            x[(1, t)] = time.sin();
        }

        // DMD without lifting
        let config_plain = DmdConfig {
            dt,
            ..Default::default()
        };
        let result_plain = dmd(&x, &config_plain).unwrap();
        assert!(result_plain.lifting_info.is_none());

        // DMD with polynomial lifting
        let config_lifted = DmdConfig {
            dt,
            lifting: Some(LiftingConfig::Polynomial { degree: 2 }),
            ..Default::default()
        };
        let result_lifted = dmd(&x, &config_lifted).unwrap();
        assert!(result_lifted.lifting_info.is_some());
        assert_eq!(result_lifted.n_vars_original(), 2);

        // Predictions should return original space dimensions
        let pred = predict_modes(&result_lifted, 10, None).unwrap();
        assert_eq!(pred.nrows(), 2); // original 2 variables, not 4 lifted
        assert_eq!(pred.ncols(), 10);
    }

    #[test]
    fn test_dmd_with_trig_lifting() {
        use crate::dmd::dmd;
        use crate::predict::predict_modes;
        use crate::types::DmdConfig;

        let n_time = 100;
        let dt = 0.1;
        let mut x = Mat::<f64>::zeros(1, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            x[(0, t)] = (2.0 * std::f64::consts::PI * 0.5 * time).sin();
        }

        let config = DmdConfig {
            dt,
            lifting: Some(LiftingConfig::Trigonometric { harmonics: 1 }),
            ..Default::default()
        };
        let result = dmd(&x, &config).unwrap();
        assert!(result.is_lifted());
        assert_eq!(result.n_vars_original(), 1);

        // data_dim should reflect lifted dimensions
        assert_eq!(result.data_dim.0, 3); // 1 * (1 + 2*1) = 3

        let pred = predict_modes(&result, 5, None).unwrap();
        assert_eq!(pred.nrows(), 1); // projected back to original 1 var
    }

    #[test]
    fn test_dmd_with_delay_lifting() {
        use crate::dmd::dmd;
        use crate::types::DmdConfig;

        let n_time = 50;
        let dt = 0.1;
        let mut x = Mat::<f64>::zeros(1, n_time);
        for t in 0..n_time {
            let time = t as f64 * dt;
            x[(0, t)] = time.sin() + 0.3 * (3.0 * time).cos();
        }

        let config = DmdConfig {
            dt,
            lifting: Some(LiftingConfig::Delay { delays: 5 }),
            ..Default::default()
        };
        let result = dmd(&x, &config).unwrap();
        assert!(result.is_lifted());
        assert_eq!(result.n_vars_original(), 1);
        // Delay reduces time points: 50 - 5 = 45
        assert_eq!(result.data_dim.1, 45);
        assert_eq!(result.data_dim.0, 6); // 1 * (5+1)
    }
}
