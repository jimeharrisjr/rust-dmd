use faer::Mat;

use crate::types::DmdError;

/// Validate that a matrix meets minimum dimension requirements and contains no NaN/Inf.
pub fn validate_matrix(x: &Mat<f64>, min_rows: usize, min_cols: usize) -> Result<(), DmdError> {
    let (rows, cols) = (x.nrows(), x.ncols());
    if rows < min_rows {
        return Err(DmdError::InvalidInput(format!(
            "matrix has {rows} rows, need at least {min_rows}"
        )));
    }
    if cols < min_cols {
        return Err(DmdError::InvalidInput(format!(
            "matrix has {cols} columns, need at least {min_cols}"
        )));
    }
    for j in 0..cols {
        for i in 0..rows {
            let val = x[(i, j)];
            if val.is_nan() || val.is_infinite() {
                return Err(DmdError::InvalidInput(
                    "matrix contains NaN or Inf values".to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// Determine truncation rank from singular values.
///
/// If `rank` is Some, clamp to valid range. Otherwise, select rank
/// capturing at least `threshold` fraction of total variance.
pub fn determine_rank(singular_values: &[f64], rank: Option<usize>, threshold: f64) -> usize {
    let n = singular_values.len();
    if n == 0 {
        return 0;
    }

    match rank {
        Some(r) => r.min(n).max(1),
        None => {
            let total: f64 = singular_values.iter().map(|s| s * s).sum();
            if total == 0.0 {
                return 1;
            }
            let mut cumulative = 0.0;
            for (i, &s) in singular_values.iter().enumerate() {
                cumulative += s * s;
                if cumulative / total >= threshold {
                    return i + 1;
                }
            }
            n
        }
    }
}

/// Compute the Moore-Penrose pseudo-inverse via SVD.
pub fn pinv(a: &Mat<f64>, tol: Option<f64>) -> Result<Mat<f64>, DmdError> {
    let svd = a.svd().map_err(|e| DmdError::SvdFailed(format!("{e:?}")))?;
    let u = svd.U();
    let s_col = svd.S().column_vector();
    let v = svd.V();

    let k = s_col.nrows();
    let max_sv = (0..k).map(|i| s_col[i].abs()).fold(0.0_f64, f64::max);

    let tol = tol.unwrap_or_else(|| {
        let max_dim = a.nrows().max(a.ncols()) as f64;
        max_sv * max_dim * f64::EPSILON
    });

    // pinv(A) = V S_inv U^T
    let m = a.nrows();
    let n = a.ncols();
    let mut result = Mat::<f64>::zeros(n, m);

    for idx in 0..k {
        let si = s_col[idx];
        if si.abs() > tol {
            let si_inv = 1.0 / si;
            // Outer product: v_col * si_inv * u_col^T
            for j in 0..n {
                for i in 0..m {
                    result[(j, i)] += v[(j, idx)] * si_inv * u[(i, idx)];
                }
            }
        }
    }

    Ok(result)
}

/// Compute row means of a matrix.
pub fn row_means(x: &Mat<f64>) -> Vec<f64> {
    let (nrows, ncols) = (x.nrows(), x.ncols());
    let mut means = vec![0.0; nrows];
    for i in 0..nrows {
        let mut sum = 0.0;
        for j in 0..ncols {
            sum += x[(i, j)];
        }
        means[i] = sum / ncols as f64;
    }
    means
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

    #[test]
    fn test_validate_matrix_ok() {
        let m = Mat::<f64>::identity(3, 3);
        assert!(validate_matrix(&m, 1, 1).is_ok());
    }

    #[test]
    fn test_validate_matrix_too_small() {
        let m = Mat::<f64>::identity(2, 2);
        assert!(validate_matrix(&m, 3, 1).is_err());
    }

    #[test]
    fn test_validate_matrix_nan() {
        let mut m = Mat::<f64>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = f64::NAN;
        assert!(validate_matrix(&m, 1, 1).is_err());
    }

    #[test]
    fn test_determine_rank_explicit() {
        let s = vec![10.0, 5.0, 1.0, 0.1];
        assert_eq!(determine_rank(&s, Some(2), 0.99), 2);
    }

    #[test]
    fn test_determine_rank_auto() {
        let s = vec![10.0, 5.0, 1.0, 0.1];
        let total: f64 = s.iter().map(|x| x * x).sum();
        let r = determine_rank(&s, None, 0.99);
        let captured: f64 = s.iter().take(r).map(|x| x * x).sum();
        assert!(captured / total >= 0.99);
    }

    #[test]
    fn test_pinv_identity() {
        let m = Mat::<f64>::identity(3, 3);
        let m_inv = pinv(&m, None).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(m_inv[(i, j)], expected, 1e-10);
            }
        }
    }

    #[test]
    fn test_pinv_rectangular() {
        let mut m = Mat::<f64>::zeros(3, 2);
        m[(0, 0)] = 1.0;
        m[(1, 1)] = 1.0;
        let m_inv = pinv(&m, None).unwrap();
        assert_eq!(m_inv.nrows(), 2);
        assert_eq!(m_inv.ncols(), 3);
        // A * pinv(A) * A should equal A
        let product = &m * &m_inv * &m;
        for i in 0..3 {
            for j in 0..2 {
                assert_near(product[(i, j)], m[(i, j)], 1e-10);
            }
        }
    }

    #[test]
    fn test_row_means() {
        let mut m = Mat::<f64>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 3.0;
        m[(1, 0)] = 2.0;
        m[(1, 1)] = 4.0;
        let means = row_means(&m);
        assert_near(means[0], 2.0, 1e-10);
        assert_near(means[1], 3.0, 1e-10);
    }
}
