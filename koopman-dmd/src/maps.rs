use std::f64::consts::PI;

use faer::Mat;

/// Trait for dynamical system maps.
///
/// A map takes a state vector and returns the next state.
pub trait MapFn: Send + Sync {
    /// Apply one iteration of the map.
    fn step(&self, state: &[f64]) -> Vec<f64>;

    /// State space dimension.
    fn dim(&self) -> usize;

    /// Map name.
    fn name(&self) -> &str;
}

/// Chirikov standard map (2D, symplectic/area-preserving).
///
/// y' = (y + ε sin(2πx)) mod 1
/// x' = (x + y') mod 1
#[derive(Debug, Clone)]
pub struct StandardMap {
    pub epsilon: f64,
}

impl Default for StandardMap {
    fn default() -> Self {
        Self { epsilon: 0.12 }
    }
}

impl MapFn for StandardMap {
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let x = state[0];
        let y = state[1];
        let y_new = (y + self.epsilon * (2.0 * PI * x).sin()).rem_euclid(1.0);
        let x_new = (x + y_new).rem_euclid(1.0);
        vec![x_new, y_new]
    }
    fn dim(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "standard_map"
    }
}

/// Froeschlé map (4D, symplectic) — two coupled standard maps.
///
/// y1' = (y1 + ε1 sin(2πx1) + η sin(2π(x1+x2))) mod 1
/// x1' = (x1 + y1') mod 1
/// y2' = (y2 + ε2 sin(2πx2) + η sin(2π(x1+x2))) mod 1
/// x2' = (x2 + y2') mod 1
#[derive(Debug, Clone)]
pub struct FroeschleMap {
    pub epsilon1: f64,
    pub epsilon2: f64,
    pub eta: f64,
}

impl Default for FroeschleMap {
    fn default() -> Self {
        Self {
            epsilon1: 0.02,
            epsilon2: 0.02,
            eta: 0.01,
        }
    }
}

impl MapFn for FroeschleMap {
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let (x1, y1, x2, y2) = (state[0], state[1], state[2], state[3]);
        let coupling = self.eta * (2.0 * PI * x1 + 2.0 * PI * x2).sin();

        let y1_new = (y1 + self.epsilon1 * (2.0 * PI * x1).sin() + coupling).rem_euclid(1.0);
        let x1_new = (x1 + y1_new).rem_euclid(1.0);
        let y2_new = (y2 + self.epsilon2 * (2.0 * PI * x2).sin() + coupling).rem_euclid(1.0);
        let x2_new = (x2 + y2_new).rem_euclid(1.0);

        vec![x1_new, y1_new, x2_new, y2_new]
    }
    fn dim(&self) -> usize {
        4
    }
    fn name(&self) -> &str {
        "froeschle_map"
    }
}

/// Extended standard map (3D, action-action-angle).
///
/// x' = (x + ε sin(2πz) + δ sin(2πy)) mod 1
/// y' = (y + ε sin(2πz)) mod 1
/// z' = (z + x') mod 1
#[derive(Debug, Clone)]
pub struct ExtendedStandardMap {
    pub epsilon: f64,
    pub delta: f64,
}

impl Default for ExtendedStandardMap {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            delta: 0.001,
        }
    }
}

impl MapFn for ExtendedStandardMap {
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let (x, y, z) = (state[0], state[1], state[2]);
        let x_new = (x + self.epsilon * (2.0 * PI * z).sin() + self.delta * (2.0 * PI * y).sin())
            .rem_euclid(1.0);
        let y_new = (y + self.epsilon * (2.0 * PI * z).sin()).rem_euclid(1.0);
        let z_new = (z + x_new).rem_euclid(1.0);
        vec![x_new, y_new, z_new]
    }
    fn dim(&self) -> usize {
        3
    }
    fn name(&self) -> &str {
        "extended_standard_map"
    }
}

/// Hénon map (2D, dissipative).
///
/// x' = 1 - a·x² + y
/// y' = b·x
#[derive(Debug, Clone)]
pub struct HenonMap {
    pub a: f64,
    pub b: f64,
}

impl Default for HenonMap {
    fn default() -> Self {
        Self { a: 1.4, b: 0.3 }
    }
}

impl MapFn for HenonMap {
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let x = state[0];
        let y = state[1];
        vec![1.0 - self.a * x * x + y, self.b * x]
    }
    fn dim(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "henon_map"
    }
}

/// Logistic map (1D, dissipative).
///
/// x' = r·x·(1-x)
#[derive(Debug, Clone)]
pub struct LogisticMap {
    pub r: f64,
}

impl Default for LogisticMap {
    fn default() -> Self {
        Self { r: 3.9 }
    }
}

impl MapFn for LogisticMap {
    fn step(&self, state: &[f64]) -> Vec<f64> {
        let x = state[0];
        vec![self.r * x * (1.0 - x)]
    }
    fn dim(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "logistic_map"
    }
}

/// A wrapper that turns a closure into a MapFn.
pub struct ClosureMap<F: Fn(&[f64]) -> Vec<f64> + Send + Sync> {
    func: F,
    dim: usize,
    name: String,
}

impl<F: Fn(&[f64]) -> Vec<f64> + Send + Sync> ClosureMap<F> {
    pub fn new(func: F, dim: usize, name: impl Into<String>) -> Self {
        Self {
            func,
            dim,
            name: name.into(),
        }
    }
}

impl<F: Fn(&[f64]) -> Vec<f64> + Send + Sync> MapFn for ClosureMap<F> {
    fn step(&self, state: &[f64]) -> Vec<f64> {
        (self.func)(state)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Generate a trajectory by iterating a map from an initial condition.
///
/// Returns a matrix (n_dim × n_iter+1) where column 0 is the initial condition.
pub fn generate_trajectory(initial_condition: &[f64], map: &dyn MapFn, n_iter: usize) -> Mat<f64> {
    let n_dim = initial_condition.len();
    let mut traj = Mat::<f64>::zeros(n_dim, n_iter + 1);

    // Store initial condition
    for i in 0..n_dim {
        traj[(i, 0)] = initial_condition[i];
    }

    let mut state = initial_condition.to_vec();
    for k in 1..=n_iter {
        state = map.step(&state);
        for i in 0..n_dim {
            traj[(i, k)] = state[i];
        }
    }

    traj
}

/// Generate a 2D grid of initial conditions.
///
/// Returns (x_coords, y_coords) vectors.
pub fn generate_phase_grid(
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
) -> (Vec<f64>, Vec<f64>) {
    let x_coords: Vec<f64> = (0..resolution)
        .map(|i| x_range.0 + (x_range.1 - x_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();
    let y_coords: Vec<f64> = (0..resolution)
        .map(|i| y_range.0 + (y_range.1 - y_range.0) * i as f64 / (resolution - 1) as f64)
        .collect();
    (x_coords, y_coords)
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
    fn test_standard_map_integrable() {
        // With epsilon=0, orbits lie on lines of constant y
        let map = StandardMap { epsilon: 0.0 };
        let state = vec![0.3, 0.2];
        let next = map.step(&state);
        assert_near(next[1], 0.2, 1e-12); // y unchanged
        assert_near(next[0], 0.5, 1e-12); // x = 0.3 + 0.2 = 0.5
    }

    #[test]
    fn test_standard_map_modular() {
        let map = StandardMap { epsilon: 0.12 };
        let state = vec![0.9, 0.9];
        let next = map.step(&state);
        assert!(next[0] >= 0.0 && next[0] < 1.0);
        assert!(next[1] >= 0.0 && next[1] < 1.0);
    }

    #[test]
    fn test_henon_map_fixed_point() {
        // The Henon map with a=0, b=0 has fixed point at (1, 0)
        let map = HenonMap { a: 0.0, b: 0.0 };
        let state = vec![0.5, 0.0];
        let next = map.step(&state);
        assert_near(next[0], 1.0, 1e-12); // 1 - 0 + 0 = 1
        assert_near(next[1], 0.0, 1e-12);
    }

    #[test]
    fn test_logistic_map_fixed_point() {
        // r=2 has fixed point at x=0.5
        let map = LogisticMap { r: 2.0 };
        let state = vec![0.5];
        let next = map.step(&state);
        assert_near(next[0], 0.5, 1e-12); // 2*0.5*0.5 = 0.5
    }

    #[test]
    fn test_logistic_map_chaos() {
        let map = LogisticMap { r: 4.0 };
        let state = vec![0.1];
        let next = map.step(&state);
        assert_near(next[0], 0.36, 1e-12); // 4*0.1*0.9 = 0.36
    }

    #[test]
    fn test_froeschle_map_uncoupled() {
        // With eta=0, subsystems are independent
        let map = FroeschleMap {
            epsilon1: 0.0,
            epsilon2: 0.0,
            eta: 0.0,
        };
        let state = vec![0.3, 0.2, 0.5, 0.1];
        let next = map.step(&state);
        // Subsystem 1: x1'=0.3+0.2=0.5, y1'=0.2
        assert_near(next[0], 0.5, 1e-12);
        assert_near(next[1], 0.2, 1e-12);
        // Subsystem 2: x2'=0.5+0.1=0.6, y2'=0.1
        assert_near(next[2], 0.6, 1e-12);
        assert_near(next[3], 0.1, 1e-12);
    }

    #[test]
    fn test_extended_standard_map() {
        let map = ExtendedStandardMap::default();
        let state = vec![0.5, 0.3, 0.2];
        let next = map.step(&state);
        assert_eq!(next.len(), 3);
        // All coordinates should be in [0, 1)
        for &v in &next {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_generate_trajectory() {
        let map = StandardMap { epsilon: 0.12 };
        let traj = generate_trajectory(&[0.5, 0.3], &map, 100);
        assert_eq!(traj.nrows(), 2);
        assert_eq!(traj.ncols(), 101); // initial + 100 steps
        assert_near(traj[(0, 0)], 0.5, 1e-12);
        assert_near(traj[(1, 0)], 0.3, 1e-12);
    }

    #[test]
    fn test_generate_phase_grid() {
        let (x, y) = generate_phase_grid((0.0, 1.0), (0.0, 1.0), 10);
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_near(x[0], 0.0, 1e-12);
        assert_near(x[9], 1.0, 1e-12);
    }

    #[test]
    fn test_closure_map() {
        let map = ClosureMap::new(|state: &[f64]| vec![state[0] * 2.0], 1, "doubling");
        assert_eq!(map.dim(), 1);
        assert_eq!(map.name(), "doubling");
        let next = map.step(&[0.25]);
        assert_near(next[0], 0.5, 1e-12);
    }
}
