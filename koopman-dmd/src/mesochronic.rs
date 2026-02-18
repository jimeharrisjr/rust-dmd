use std::f64::consts::PI;

use rayon::prelude::*;

use crate::harmonic::{hta_from_values, Observable};
use crate::maps::MapFn;
use crate::types::{DmdError, C64};

/// Result of a mesochronic harmonic plot computation.
#[derive(Debug, Clone)]
pub struct MhpResult {
    /// |HTA| magnitudes over the grid (resolution × resolution, row-major).
    pub hta_matrix: Vec<Vec<f64>>,
    /// arg(HTA) phases over the grid.
    pub phase_matrix: Vec<Vec<f64>>,
    /// X-axis grid coordinates.
    pub x_coords: Vec<f64>,
    /// Y-axis grid coordinates.
    pub y_coords: Vec<f64>,
    /// Frequency parameter.
    pub omega: f64,
    /// Period = 1/omega.
    pub period: f64,
    /// Number of iterations per grid point.
    pub n_iter: usize,
    /// Observable used.
    pub observable: String,
}

/// Compute a Mesochronic Harmonic Plot over a 2D grid.
///
/// For each initial condition (x, y) in the grid, generates a trajectory,
/// evaluates the observable, and computes the HTA at frequency omega.
///
/// Computation is parallelized over rows using rayon.
///
/// # Arguments
/// * `map` - Dynamical system map (must be 2D or higher).
/// * `x_range` - (x_min, x_max) range.
/// * `y_range` - (y_min, y_max) range.
/// * `resolution` - Grid points per dimension.
/// * `observable` - Observable function.
/// * `omega` - Frequency parameter (use 1/period).
/// * `n_iter` - Iterations per grid point.
pub fn mesochronic_compute(
    map: &(dyn MapFn + Sync),
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
    observable: &Observable,
    omega: f64,
    n_iter: usize,
) -> Result<MhpResult, DmdError> {
    if resolution < 2 {
        return Err(DmdError::InvalidInput(
            "resolution must be at least 2".into(),
        ));
    }
    if n_iter == 0 {
        return Err(DmdError::InvalidInput("n_iter must be positive".into()));
    }

    let x_coords: Vec<f64> = (0..resolution)
        .map(|i| x_range.0 + (x_range.1 - x_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();
    let y_coords: Vec<f64> = (0..resolution)
        .map(|i| y_range.0 + (y_range.1 - y_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();

    // Pre-compute phase factors
    let phase_factors: Vec<C64> = (0..n_iter)
        .map(|k| {
            let phase = 2.0 * PI * k as f64 * omega;
            C64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Parallel computation over rows
    let obs = *observable;
    let results: Vec<Vec<(f64, f64)>> = x_coords
        .par_iter()
        .map(|&xi| {
            let mut row = Vec::with_capacity(resolution);
            for &yj in &y_coords {
                let mut state = vec![xi, yj];
                // Pad state to map dimension if needed
                while state.len() < map.dim() {
                    state.push(0.0);
                }

                let mut f_values = Vec::with_capacity(n_iter);
                for _ in 0..n_iter {
                    f_values.push(obs.eval(&state));
                    state = map.step(&state);
                }

                // Compute HTA using pre-computed phase factors
                let mut sum = C64::zero();
                for k in 0..n_iter {
                    sum += phase_factors[k] * C64::new(f_values[k], 0.0);
                }
                let hta = sum / n_iter as f64;
                row.push((hta.norm(), hta.arg()));
            }
            row
        })
        .collect();

    // Unpack into matrices
    let mut hta_matrix = vec![vec![0.0; resolution]; resolution];
    let mut phase_matrix = vec![vec![0.0; resolution]; resolution];

    for (i, row) in results.iter().enumerate() {
        for (j, &(mag, phase)) in row.iter().enumerate() {
            hta_matrix[i][j] = mag;
            phase_matrix[i][j] = phase;
        }
    }

    Ok(MhpResult {
        hta_matrix,
        phase_matrix,
        x_coords,
        y_coords,
        omega,
        period: 1.0 / omega,
        n_iter,
        observable: observable.name().to_string(),
    })
}

/// Result of a mesochronic scatter plot computation.
#[derive(Debug, Clone)]
pub struct MhspResult {
    /// |HTA| values for each observable, each Vec has resolution² entries.
    pub hta_lists: Vec<Vec<f64>>,
    /// Observable names.
    pub observable_names: Vec<String>,
    /// Initial conditions (flattened: 2 × n_points).
    pub initial_conditions_x: Vec<f64>,
    pub initial_conditions_y: Vec<f64>,
    /// Frequency and period.
    pub omega: f64,
    pub period: f64,
    /// Iterations per point.
    pub n_iter: usize,
}

/// Compute HTA values for multiple observables over a grid (for scatter plots).
///
/// # Arguments
/// * `map` - Dynamical system map.
/// * `x_range`, `y_range` - Grid ranges.
/// * `resolution` - Grid points per dimension.
/// * `observables` - Slice of observables to evaluate.
/// * `omega` - Frequency parameter.
/// * `n_iter` - Iterations per grid point.
pub fn mesochronic_scatter(
    map: &(dyn MapFn + Sync),
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
    observables: &[Observable],
    omega: f64,
    n_iter: usize,
) -> Result<MhspResult, DmdError> {
    if observables.is_empty() {
        return Err(DmdError::InvalidInput(
            "need at least one observable".into(),
        ));
    }

    let x_coords: Vec<f64> = (0..resolution)
        .map(|i| x_range.0 + (x_range.1 - x_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();
    let y_coords: Vec<f64> = (0..resolution)
        .map(|i| y_range.0 + (y_range.1 - y_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();

    let n_obs = observables.len();
    let n_points = resolution * resolution;

    // Build flat list of initial conditions
    let mut ics_x = Vec::with_capacity(n_points);
    let mut ics_y = Vec::with_capacity(n_points);
    for &xi in &x_coords {
        for &yj in &y_coords {
            ics_x.push(xi);
            ics_y.push(yj);
        }
    }

    let obs_copy: Vec<Observable> = observables.to_vec();

    // Parallel computation over grid points
    let results: Vec<Vec<f64>> = (0..n_points)
        .into_par_iter()
        .map(|idx| {
            let mut state = vec![ics_x[idx], ics_y[idx]];
            while state.len() < map.dim() {
                state.push(0.0);
            }

            // Generate trajectory and collect all observable values
            let mut obs_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_iter); n_obs];
            for _ in 0..n_iter {
                for (oi, obs) in obs_copy.iter().enumerate() {
                    obs_values[oi].push(obs.eval(&state));
                }
                state = map.step(&state);
            }

            // Compute HTA for each observable
            obs_values
                .iter()
                .map(|vals| hta_from_values(vals, omega).norm())
                .collect()
        })
        .collect();

    // Transpose: results[point][obs] -> hta_lists[obs][point]
    let mut hta_lists = vec![vec![0.0; n_points]; n_obs];
    for (pt, obs_mags) in results.iter().enumerate() {
        for (oi, &mag) in obs_mags.iter().enumerate() {
            hta_lists[oi][pt] = mag;
        }
    }

    let observable_names = observables.iter().map(|o| o.name().to_string()).collect();

    Ok(MhspResult {
        hta_lists,
        observable_names,
        initial_conditions_x: ics_x,
        initial_conditions_y: ics_y,
        omega,
        period: 1.0 / omega,
        n_iter,
    })
}

/// Compute a 2D section MHP for higher-dimensional systems.
///
/// Varies two dimensions while fixing the others, then computes HTA at each point.
///
/// # Arguments
/// * `map` - Higher-dimensional dynamical system map.
/// * `section_dims` - (dim1, dim2) indices of dimensions to vary (0-indexed).
/// * `fixed_values` - Values for all dimensions (varied dims will be overwritten).
/// * `x_range`, `y_range` - Ranges for the two varied dimensions.
/// * `resolution` - Grid resolution.
/// * `observable` - Observable function.
/// * `omega` - Frequency parameter.
/// * `n_iter` - Iterations per point.
#[allow(clippy::too_many_arguments)]
pub fn mesochronic_section(
    map: &(dyn MapFn + Sync),
    section_dims: (usize, usize),
    fixed_values: &[f64],
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
    observable: &Observable,
    omega: f64,
    n_iter: usize,
) -> Result<MhpResult, DmdError> {
    let total_dim = map.dim();
    if fixed_values.len() != total_dim {
        return Err(DmdError::InvalidInput(format!(
            "fixed_values has length {}, expected {} (map dimension)",
            fixed_values.len(),
            total_dim
        )));
    }
    if section_dims.0 >= total_dim || section_dims.1 >= total_dim {
        return Err(DmdError::InvalidInput("section_dims out of range".into()));
    }

    let x_coords: Vec<f64> = (0..resolution)
        .map(|i| x_range.0 + (x_range.1 - x_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();
    let y_coords: Vec<f64> = (0..resolution)
        .map(|i| y_range.0 + (y_range.1 - y_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();

    let obs = *observable;
    let fixed = fixed_values.to_vec();
    let (d1, d2) = section_dims;

    let results: Vec<Vec<(f64, f64)>> = x_coords
        .par_iter()
        .map(|&xi| {
            let mut row = Vec::with_capacity(resolution);
            for &yj in &y_coords {
                let mut state = fixed.clone();
                state[d1] = xi;
                state[d2] = yj;

                let mut f_values = Vec::with_capacity(n_iter);
                for _ in 0..n_iter {
                    f_values.push(obs.eval(&state));
                    state = map.step(&state);
                }

                let hta = hta_from_values(&f_values, omega);
                row.push((hta.norm(), hta.arg()));
            }
            row
        })
        .collect();

    let mut hta_matrix = vec![vec![0.0; resolution]; resolution];
    let mut phase_matrix = vec![vec![0.0; resolution]; resolution];
    for (i, row) in results.iter().enumerate() {
        for (j, &(mag, phase)) in row.iter().enumerate() {
            hta_matrix[i][j] = mag;
            phase_matrix[i][j] = phase;
        }
    }

    Ok(MhpResult {
        hta_matrix,
        phase_matrix,
        x_coords,
        y_coords,
        omega,
        period: 1.0 / omega,
        n_iter,
        observable: observable.name().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maps::{FroeschleMap, StandardMap};

    #[test]
    fn test_mesochronic_compute_basic() {
        let map = StandardMap { epsilon: 0.12 };
        let result = mesochronic_compute(
            &map,
            (0.0, 1.0),
            (0.0, 1.0),
            10, // small grid for test speed
            &Observable::SinPi,
            0.5, // period-2
            1000,
        )
        .unwrap();

        assert_eq!(result.hta_matrix.len(), 10);
        assert_eq!(result.hta_matrix[0].len(), 10);
        assert_eq!(result.x_coords.len(), 10);
        assert_eq!(result.y_coords.len(), 10);
        assert_eq!(result.omega, 0.5);
        assert_eq!(result.period, 2.0);

        // All magnitudes should be non-negative
        for row in &result.hta_matrix {
            for &v in row {
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn test_mesochronic_has_structure() {
        // With epsilon=0.12 the standard map has mixed dynamics.
        // We should see variation in HTA magnitudes across the grid.
        let map = StandardMap { epsilon: 0.12 };
        let result = mesochronic_compute(
            &map,
            (0.0, 1.0),
            (0.0, 1.0),
            5,
            &Observable::SinPi,
            0.5,
            5000,
        )
        .unwrap();

        let mut min_hta = f64::INFINITY;
        let mut max_hta = 0.0_f64;
        for row in &result.hta_matrix {
            for &v in row {
                min_hta = min_hta.min(v);
                max_hta = max_hta.max(v);
            }
        }
        // There should be some spread in HTA values
        assert!(max_hta > min_hta);
    }

    #[test]
    fn test_mesochronic_scatter() {
        let map = StandardMap { epsilon: 0.12 };
        let result = mesochronic_scatter(
            &map,
            (0.0, 1.0),
            (0.0, 1.0),
            5,
            &[Observable::SinPi, Observable::SinPiXY],
            0.5,
            1000,
        )
        .unwrap();

        assert_eq!(result.hta_lists.len(), 2);
        assert_eq!(result.hta_lists[0].len(), 25); // 5x5 grid
        assert_eq!(result.observable_names.len(), 2);
        assert_eq!(result.observable_names[0], "sin_pi");
        assert_eq!(result.observable_names[1], "sin_pi_xy");
    }

    #[test]
    fn test_mesochronic_section() {
        let map = FroeschleMap::default();
        let result = mesochronic_section(
            &map,
            (0, 1),                // vary x1, y1
            &[0.0, 0.0, 0.5, 0.3], // fixed values for all 4 dims
            (0.0, 1.0),
            (0.0, 1.0),
            5,
            &Observable::SinPi,
            0.5,
            1000,
        )
        .unwrap();

        assert_eq!(result.hta_matrix.len(), 5);
        assert_eq!(result.hta_matrix[0].len(), 5);
    }

    #[test]
    fn test_mesochronic_error_cases() {
        let map = StandardMap::default();
        assert!(mesochronic_compute(
            &map,
            (0.0, 1.0),
            (0.0, 1.0),
            1,
            &Observable::SinPi,
            0.5,
            100
        )
        .is_err());

        assert!(
            mesochronic_compute(&map, (0.0, 1.0), (0.0, 1.0), 5, &Observable::SinPi, 0.5, 0)
                .is_err()
        );
    }
}
