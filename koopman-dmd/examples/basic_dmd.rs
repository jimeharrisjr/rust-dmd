//! Basic DMD example: decompose an oscillatory signal.

use koopman_dmd::{dmd, dmd_spectrum, dmd_stability, predict_modes, DmdConfig};

fn main() {
    // Create a 2-variable oscillating signal
    let n = 200;
    let mut data = faer::Mat::<f64>::zeros(2, n);
    for j in 0..n {
        let t = j as f64 * 0.05;
        data[(0, j)] = t.sin() + 0.3 * (3.0 * t).sin();
        data[(1, j)] = t.cos() + 0.3 * (3.0 * t).cos();
    }

    // Compute DMD
    let config = DmdConfig::default();
    let result = dmd(&data, &config).unwrap();

    println!("DMD Decomposition");
    println!("  Rank: {}", result.rank);
    println!(
        "  Data: {} vars x {} time steps",
        result.data_dim.0, result.data_dim.1
    );

    // Eigenvalue spectrum
    let spec = dmd_spectrum(&result, config.dt);
    println!("\nEigenvalue Spectrum:");
    for m in &spec {
        println!(
            "  Mode {}: |λ|={:.4}, freq={:.4}, stability={}",
            m.index, m.magnitude, m.frequency, m.stability
        );
    }

    // Stability analysis
    let stab = dmd_stability(&result, 1e-6);
    println!("\nStability:");
    println!("  Stable: {}", stab.is_stable);
    println!("  Spectral radius: {:.6}", stab.spectral_radius);

    // Predict future
    let pred = predict_modes(&result, 20, None).unwrap();
    println!("\nPredicted 20 steps ahead:");
    println!("  x[0]: {:.4} -> {:.4}", pred[(0, 0)], pred[(0, 19)]);
    println!("  x[1]: {:.4} -> {:.4}", pred[(1, 0)], pred[(1, 19)]);
}
