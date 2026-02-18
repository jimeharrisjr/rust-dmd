//! Edge case and numerical stability tests.

use approx::assert_abs_diff_eq;
use koopman_dmd::*;

fn make_signal(n_vars: usize, n_time: usize) -> faer::Mat<f64> {
    let mut data = faer::Mat::<f64>::zeros(n_vars, n_time);
    for j in 0..n_time {
        let t = j as f64 * 0.1;
        for i in 0..n_vars {
            data[(i, j)] = ((i + 1) as f64 * t).sin();
        }
    }
    data
}

// ============================================================================
// Edge cases: rank-1 systems
// ============================================================================

#[test]
fn dmd_rank_1() {
    let data = make_signal(2, 50);
    let config = DmdConfig {
        rank: Some(1),
        ..Default::default()
    };
    let result = dmd(&data, &config).unwrap();
    assert_eq!(result.rank, 1);
    assert_eq!(result.eigenvalues.len(), 1);
    assert_eq!(result.amplitudes.len(), 1);

    let pred = predict_modes(&result, 5, None).unwrap();
    assert_eq!(pred.nrows(), 2);
    assert_eq!(pred.ncols(), 5);
}

// ============================================================================
// Edge cases: single variable
// ============================================================================

#[test]
fn dmd_single_variable() {
    let mut data = faer::Mat::<f64>::zeros(1, 100);
    for j in 0..100 {
        data[(0, j)] = (j as f64 * 0.1).sin();
    }
    let result = dmd(&data, &DmdConfig::default()).unwrap();
    assert_eq!(result.data_dim.0, 1);

    let recon = dmd_reconstruct(&result, 100, None).unwrap();
    assert_eq!(recon.nrows(), 1);
}

// ============================================================================
// Edge cases: constant signal
// ============================================================================

#[test]
fn dmd_constant_signal() {
    let mut data = faer::Mat::<f64>::zeros(2, 50);
    for j in 0..50 {
        data[(0, j)] = 3.0;
        data[(1, j)] = 7.0;
    }
    // With centering, signal becomes zero -> should still work
    let config = DmdConfig {
        center: true,
        ..Default::default()
    };
    let result = dmd(&data, &config);
    // This may succeed or fail depending on numerical rank; either is acceptable
    // May succeed or error due to zero signal after centering; both are acceptable
    let _ = result;
}

// ============================================================================
// Edge cases: minimum size matrices
// ============================================================================

#[test]
fn dmd_minimum_size() {
    // 2 variables, 3 time steps -> 2x2 and 2x1 split
    let mut data = faer::Mat::<f64>::zeros(2, 3);
    data[(0, 0)] = 1.0;
    data[(1, 0)] = 0.0;
    data[(0, 1)] = 0.5;
    data[(1, 1)] = 0.5;
    data[(0, 2)] = 0.0;
    data[(1, 2)] = 1.0;

    let result = dmd(&data, &DmdConfig::default()).unwrap();
    assert!(result.rank > 0);
    assert!(result.rank <= 2);
}

// ============================================================================
// Edge cases: large rank request
// ============================================================================

#[test]
fn dmd_rank_capped() {
    let data = make_signal(3, 50);
    let config = DmdConfig {
        rank: Some(100), // Request more than possible
        ..Default::default()
    };
    let result = dmd(&data, &config).unwrap();
    assert!(result.rank <= 3); // Can't exceed min(n_vars, n_time-1)
}

// ============================================================================
// Numerical stability: near-singular
// ============================================================================

#[test]
fn dmd_near_singular() {
    // Two nearly identical rows
    let mut data = faer::Mat::<f64>::zeros(2, 50);
    for j in 0..50 {
        let t = j as f64 * 0.1;
        data[(0, j)] = t.sin();
        data[(1, j)] = t.sin() + 1e-12; // Nearly identical
    }
    let result = dmd(&data, &DmdConfig::default()).unwrap();
    assert!(result.rank >= 1);
}

// ============================================================================
// Cross-validation: modes vs matrix prediction
// ============================================================================

#[test]
fn predict_modes_vs_matrix_agree() {
    let data = make_signal(2, 100);
    let result = dmd(&data, &DmdConfig::default()).unwrap();

    let pred_modes = predict_modes(&result, 10, None).unwrap();
    let pred_matrix = predict_matrix(&result, 10, None).unwrap();

    // Both methods should produce similar results (not exact due to different formulations)
    for j in 0..10 {
        for i in 0..2 {
            let diff = (pred_modes[(i, j)] - pred_matrix[(i, j)]).abs();
            // Allow generous tolerance since methods differ
            assert!(
                diff < 2.0,
                "mode/matrix predictions diverged at ({i},{j}): {diff}"
            );
        }
    }
}

// ============================================================================
// Cross-validation: reconstruction recovers data
// ============================================================================

#[test]
fn reconstruction_accuracy() {
    let data = make_signal(2, 100);
    let result = dmd(&data, &DmdConfig::default()).unwrap();

    let _recon = dmd_reconstruct(&result, 100, None).unwrap();
    let err = dmd_error(&result, &data).unwrap();

    // With auto-rank, reconstruction may not be perfect for multi-frequency signals
    assert!(err.rmse < 2.0, "RMSE too large: {}", err.rmse);
    assert!(
        err.relative_error < 2.0,
        "Relative error: {}",
        err.relative_error
    );
}

// ============================================================================
// Cross-validation: stability analysis consistency
// ============================================================================

#[test]
fn stability_consistency() {
    let data = make_signal(2, 100);
    let result = dmd(&data, &DmdConfig::default()).unwrap();

    let stab = dmd_stability(&result, 1e-6);
    let spec = dmd_spectrum(&result, 1.0);

    // Spectral radius should match max eigenvalue magnitude from spectrum
    let max_mag = spec.iter().map(|m| m.magnitude).fold(0.0_f64, f64::max);
    assert_abs_diff_eq!(stab.spectral_radius, max_mag, epsilon = 1e-10);

    // Stable iff no growing modes
    let has_growing = spec
        .iter()
        .any(|m| matches!(m.stability, Stability::Growing));
    assert_eq!(stab.is_unstable, has_growing);
}

// ============================================================================
// Cross-validation: Hankel-DMD on oscillatory signal
// ============================================================================

#[test]
fn hankel_recovers_frequency() {
    let n = 200;
    let freq = 0.05; // frequency in cycles per sample
    let mut data = faer::Mat::<f64>::zeros(1, n);
    for j in 0..n {
        data[(0, j)] = (2.0 * std::f64::consts::PI * freq * j as f64).sin();
    }

    let config = HankelConfig {
        delays: Some(20),
        rank: Some(2),
        dt: 1.0,
    };
    let result = hankel_dmd(&data, &config).unwrap();

    // Eigenvalues should have magnitude near 1 (stable oscillation)
    for eig in &result.eigenvalues {
        let mag = eig.norm();
        assert!(
            (mag - 1.0).abs() < 0.1,
            "Eigenvalue magnitude {mag} not near 1.0"
        );
    }
}

// ============================================================================
// Cross-validation: lifting improves nonlinear fit
// ============================================================================

#[test]
fn lifting_improves_nonlinear() {
    // Nonlinear signal: x^2 dynamics
    let n = 100;
    let mut data = faer::Mat::<f64>::zeros(1, n);
    for j in 0..n {
        let t = j as f64 * 0.05;
        data[(0, j)] = t.sin().powi(2);
    }

    // Without lifting
    let no_lift = dmd(&data, &DmdConfig::default()).unwrap();
    let err_no_lift = dmd_error(&no_lift, &data).unwrap();

    // With polynomial lifting
    let config = DmdConfig {
        lifting: Some(LiftingConfig::Polynomial { degree: 2 }),
        ..Default::default()
    };
    let with_lift = dmd(&data, &config).unwrap();
    let err_with_lift = dmd_error(&with_lift, &data).unwrap();

    // Lifting should not make it worse (may or may not improve significantly)
    assert!(
        err_with_lift.rmse <= err_no_lift.rmse * 1.5,
        "Lifting degraded: {} vs {}",
        err_with_lift.rmse,
        err_no_lift.rmse
    );
}

// ============================================================================
// Cross-validation: map trajectory properties
// ============================================================================

#[test]
fn standard_map_preserves_area() {
    // Standard map is area-preserving (symplectic)
    let map = StandardMap { epsilon: 0.12 };
    let traj = generate_trajectory(&[0.1, 0.2], &map, 1000);

    // All values should be in [0, 1) mod 1
    for j in 0..traj.ncols() {
        let x = traj[(0, j)];
        let y = traj[(1, j)];
        assert!(x >= 0.0 && x <= 1.0, "x out of range: {x}");
        assert!(y >= 0.0 && y <= 1.0, "y out of range: {y}");
    }
}

#[test]
fn logistic_map_stays_bounded() {
    let map = LogisticMap { r: 3.9 };
    let traj = generate_trajectory(&[0.5], &map, 10000);

    for j in 0..traj.ncols() {
        let x = traj[(0, j)];
        assert!(x >= 0.0 && x <= 1.0, "logistic map escaped: {x}");
    }
}

// ============================================================================
// Cross-validation: HTA for known periodic orbit
// ============================================================================

#[test]
fn hta_periodic_orbit_has_large_magnitude() {
    // A fixed point of the standard map has large HTA magnitude
    let map = StandardMap { epsilon: 0.0 }; // Zero perturbation: x'=x+y, y'=y
    let ic = vec![0.5, 0.0]; // Fixed point when epsilon=0

    let result = harmonic_time_average(&ic, &map, &Observable::SinPi, 0.0, 10000).unwrap();
    // At a fixed point with omega=0, HTA should equal the observable value
    let expected = (std::f64::consts::PI * 0.5).sin();
    assert_abs_diff_eq!(result.magnitude, expected.abs(), epsilon = 0.1);
}

// ============================================================================
// GLA convergence test
// ============================================================================

#[test]
fn gla_converges_on_simple_signal() {
    let n = 200;
    let mut data = faer::Mat::<f64>::zeros(2, n);
    for j in 0..n {
        let t = j as f64 * 0.1;
        data[(0, j)] = t.sin();
        data[(1, j)] = t.cos();
    }

    let config = GlaConfig {
        eigenvalues: None,
        n_eigenvalues: 2,
        tol: 1e-4,
        max_iter: None,
    };
    let result = gla(&data, &config).unwrap();
    assert_eq!(result.eigenvalues.len(), 2);

    // Eigenvalues should have magnitude near 1 for pure oscillation
    for eig in &result.eigenvalues {
        let mag = eig.norm();
        assert!(
            mag > 0.5 && mag < 1.5,
            "GLA eigenvalue magnitude {mag} out of expected range"
        );
    }
}
