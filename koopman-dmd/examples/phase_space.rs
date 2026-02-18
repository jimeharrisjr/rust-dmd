//! Phase space analysis example: HTA and mesochronic computation.

use koopman_dmd::{
    classify_phase_space, generate_trajectory, harmonic_time_average, mesochronic_compute,
    Observable, PhaseSpaceClass, StandardMap,
};

fn main() {
    let map = StandardMap { epsilon: 0.3 };

    // Generate a trajectory
    let traj = generate_trajectory(&[0.1, 0.2], &map, 10000);
    println!("Standard map trajectory: {} steps", traj.ncols());

    // Compute HTA for this trajectory
    let hta = harmonic_time_average(&[0.1, 0.2], &map, &Observable::SinPi, 0.1, 10000).unwrap();
    println!("\nHTA at (0.1, 0.2):");
    println!("  Magnitude: {:.6}", hta.magnitude);
    println!("  Phase: {:.4} rad", hta.phase);

    // Classify this point
    let class = classify_phase_space(&[hta.magnitude], 0.01, 0.0001);
    let label = match class[0] {
        PhaseSpaceClass::Resonating => "Resonating",
        PhaseSpaceClass::Chaotic => "Chaotic",
        PhaseSpaceClass::NonResonating => "Non-Resonating",
    };
    println!("  Classification: {label}");

    // Mesochronic computation over a small grid
    println!("\nComputing mesochronic grid (20x20)...");
    let mhp = mesochronic_compute(
        &map,
        (0.0, 1.0),
        (0.0, 1.0),
        20,
        &Observable::SinPi,
        0.1,
        5000,
    )
    .unwrap();

    // Summarize
    let mut hta_vals: Vec<f64> = Vec::new();
    for row in &mhp.hta_matrix {
        for &val in row {
            hta_vals.push(val);
        }
    }
    let min_hta = hta_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_hta = hta_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  HTA range: [{:.6}, {:.6}]", min_hta, max_hta);

    // Classify all grid points
    let classes = classify_phase_space(&hta_vals, 0.01, 0.0001);
    let n_res = classes
        .iter()
        .filter(|&&c| c == PhaseSpaceClass::Resonating)
        .count();
    let n_cha = classes
        .iter()
        .filter(|&&c| c == PhaseSpaceClass::Chaotic)
        .count();
    let n_non = classes
        .iter()
        .filter(|&&c| c == PhaseSpaceClass::NonResonating)
        .count();
    println!(
        "  Classification: {} resonating, {} chaotic, {} non-resonating",
        n_res, n_cha, n_non
    );
}
