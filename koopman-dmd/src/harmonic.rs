use std::f64::consts::PI;

use crate::maps::MapFn;
use crate::types::{DmdError, C64};

/// Built-in observable functions for HTA computation.
#[derive(Debug, Clone, Copy)]
pub enum Observable {
    /// First coordinate: x\[0\]
    Identity,
    /// sin(π·x\[0\])
    SinPi,
    /// cos(π·x\[0\])
    CosPi,
    /// sin(π·x\[0\])·sin(π·x\[1\])
    SinPiXY,
    /// cos(π·x\[0\])·cos(π·x\[1\])
    CosPiXY,
    /// sin(2π·x\[0\])
    Sin2Pi,
    /// cos(2π·x\[0\])
    Cos2Pi,
    /// sin(3π·x\[0\])·sin(17π·x\[1\])
    TrigProduct,
}

impl Observable {
    /// Evaluate the observable on a state vector.
    pub fn eval(&self, state: &[f64]) -> f64 {
        match self {
            Observable::Identity => state[0],
            Observable::SinPi => (PI * state[0]).sin(),
            Observable::CosPi => (PI * state[0]).cos(),
            Observable::SinPiXY => (PI * state[0]).sin() * (PI * state[1]).sin(),
            Observable::CosPiXY => (PI * state[0]).cos() * (PI * state[1]).cos(),
            Observable::Sin2Pi => (2.0 * PI * state[0]).sin(),
            Observable::Cos2Pi => (2.0 * PI * state[0]).cos(),
            Observable::TrigProduct => (3.0 * PI * state[0]).sin() * (17.0 * PI * state[0]).sin(),
        }
    }

    /// Name of the observable.
    pub fn name(&self) -> &str {
        match self {
            Observable::Identity => "identity",
            Observable::SinPi => "sin_pi",
            Observable::CosPi => "cos_pi",
            Observable::SinPiXY => "sin_pi_xy",
            Observable::CosPiXY => "cos_pi_xy",
            Observable::Sin2Pi => "sin_2pi",
            Observable::Cos2Pi => "cos_2pi",
            Observable::TrigProduct => "trig_product",
        }
    }
}

/// Result of a Harmonic Time Average computation.
#[derive(Debug, Clone)]
pub struct HtaResult {
    /// Complex HTA value.
    pub hta: C64,
    /// |HTA| magnitude.
    pub magnitude: f64,
    /// arg(HTA) phase.
    pub phase: f64,
    /// Frequency parameter ω ∈ (0, 0.5).
    pub omega: f64,
    /// Period = 1/ω.
    pub period: f64,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Observable name.
    pub observable: String,
}

/// Compute the Harmonic Time Average for a trajectory.
///
/// f*_ω(x) = (1/T) Σ_{k=0}^{T-1} exp(i·2π·k·ω) · f(T^k x)
///
/// # Interpretation of |HTA|
/// - ~10⁻¹: Resonating periodic set at frequency ω
/// - ~10⁻³: Chaotic region
/// - ~10⁻⁵: Non-resonating periodic set
///
/// # Arguments
/// * `initial_condition` - Starting state.
/// * `map` - Dynamical system map.
/// * `observable` - Observable function to evaluate.
/// * `omega` - Frequency parameter (0, 0.5). Use 1/period.
/// * `n_iter` - Number of iterations.
pub fn harmonic_time_average(
    initial_condition: &[f64],
    map: &dyn MapFn,
    observable: &Observable,
    omega: f64,
    n_iter: usize,
) -> Result<HtaResult, DmdError> {
    if n_iter == 0 {
        return Err(DmdError::InvalidInput("n_iter must be positive".into()));
    }

    let mut state = initial_condition.to_vec();
    let mut sum = C64::zero();

    for k in 0..n_iter {
        let f_val = observable.eval(&state);
        let phase = 2.0 * PI * k as f64 * omega;
        let weight = C64::new(phase.cos(), phase.sin());
        sum += weight * C64::new(f_val, 0.0);
        state = map.step(&state);
    }

    let hta = sum / n_iter as f64;
    let magnitude = hta.norm();
    let phase = hta.arg();

    Ok(HtaResult {
        hta,
        magnitude,
        phase,
        omega,
        period: 1.0 / omega,
        n_iter,
        observable: observable.name().to_string(),
    })
}

/// Compute HTA directly from pre-computed observable values along a trajectory.
///
/// This is more efficient when the trajectory and observable values are already available.
pub fn hta_from_values(f_values: &[f64], omega: f64) -> C64 {
    let n = f_values.len();
    if n == 0 {
        return C64::zero();
    }
    let mut sum = C64::zero();
    for k in 0..n {
        let phase = 2.0 * PI * k as f64 * omega;
        let weight = C64::new(phase.cos(), phase.sin());
        sum += weight * C64::new(f_values[k], 0.0);
    }
    sum / n as f64
}

/// Result of HTA convergence analysis.
#[derive(Debug, Clone)]
pub struct HtaConvergenceResult {
    /// Sample times at which HTA was computed.
    pub times: Vec<usize>,
    /// |HTA| at each sample time.
    pub hta_magnitudes: Vec<f64>,
    /// Estimated convergence rate (log-log slope).
    pub convergence_rate: Option<f64>,
    /// Dynamics classification.
    pub dynamics_type: DynamicsType,
    /// Frequency parameter.
    pub omega: f64,
    /// Period.
    pub period: f64,
}

/// Classification of dynamical behavior from HTA convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicsType {
    /// |HTA| converges to non-zero value (rate > -0.3).
    ResonatingPeriodic,
    /// |HTA| decays as ~1/√t (rate ~ -0.5).
    Chaotic,
    /// |HTA| decays as ~1/t (rate ~ -1).
    NonResonatingPeriodic,
    /// Could not reliably classify.
    Indeterminate,
}

impl std::fmt::Display for DynamicsType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynamicsType::ResonatingPeriodic => write!(f, "resonating_periodic"),
            DynamicsType::Chaotic => write!(f, "chaotic"),
            DynamicsType::NonResonatingPeriodic => write!(f, "non_resonating_periodic"),
            DynamicsType::Indeterminate => write!(f, "indeterminate"),
        }
    }
}

/// Analyze HTA convergence over time to classify dynamics.
///
/// Convergence rates:
/// - Periodic orbits: |HTA(t) - HTA(∞)| ~ C/t (slope ~ -1)
/// - Chaotic regions: |HTA(t)| ~ C/√t (slope ~ -0.5)
///
/// # Arguments
/// * `initial_condition` - Starting state.
/// * `map` - Dynamical system map.
/// * `observable` - Observable function.
/// * `omega` - Frequency parameter.
/// * `n_iter` - Total number of iterations.
/// * `sample_times` - Optional times at which to sample. If None, log-spaced.
pub fn hta_convergence(
    initial_condition: &[f64],
    map: &dyn MapFn,
    observable: &Observable,
    omega: f64,
    n_iter: usize,
    sample_times: Option<&[usize]>,
) -> Result<HtaConvergenceResult, DmdError> {
    if n_iter < 100 {
        return Err(DmdError::InvalidInput(
            "need at least 100 iterations for convergence analysis".into(),
        ));
    }

    // Generate trajectory and compute observable values
    let mut state = initial_condition.to_vec();
    let mut f_values = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        f_values.push(observable.eval(&state));
        state = map.step(&state);
    }

    // Determine sample times
    let times: Vec<usize> = match sample_times {
        Some(t) => t.iter().copied().filter(|&t| t <= n_iter).collect(),
        None => {
            let n_samples = ((n_iter as f64).log10() * 5.0).floor() as usize;
            let n_samples = n_samples.min(20).max(5);
            let log_min = 2.0_f64; // 100
            let log_max = (n_iter as f64).log10();
            let mut times = Vec::new();
            for i in 0..n_samples {
                let log_t = log_min + (log_max - log_min) * i as f64 / (n_samples - 1) as f64;
                let t = 10.0_f64.powf(log_t).round() as usize;
                if t <= n_iter && (times.is_empty() || *times.last().unwrap() != t) {
                    times.push(t);
                }
            }
            times
        }
    };

    // Compute HTA at each sample time
    let mut hta_magnitudes = Vec::with_capacity(times.len());
    for &t in &times {
        let hta = hta_from_values(&f_values[..t], omega);
        hta_magnitudes.push(hta.norm());
    }

    // Estimate convergence rate via log-log regression
    // Use later samples (t >= 1000) for better estimate
    let use_idx: Vec<usize> = times
        .iter()
        .enumerate()
        .filter(|(_, &t)| t >= 1000)
        .map(|(i, _)| i)
        .collect();

    let use_idx = if use_idx.len() < 3 {
        (0..times.len()).collect()
    } else {
        use_idx
    };

    let convergence_rate = if use_idx.len() >= 2 {
        let log_t: Vec<f64> = use_idx.iter().map(|&i| (times[i] as f64).log10()).collect();
        let log_hta: Vec<f64> = use_idx
            .iter()
            .map(|&i| (hta_magnitudes[i] + 1e-15).log10())
            .collect();

        // Linear regression
        let n = log_t.len() as f64;
        let sum_x: f64 = log_t.iter().sum();
        let sum_y: f64 = log_hta.iter().sum();
        let sum_x2: f64 = log_t.iter().map(|x| x * x).sum();
        let sum_xy: f64 = log_t.iter().zip(&log_hta).map(|(x, y)| x * y).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() > 1e-14 {
            Some((n * sum_xy - sum_x * sum_y) / denom)
        } else {
            None
        }
    } else {
        None
    };

    let dynamics_type = match convergence_rate {
        Some(rate) => {
            if rate > -0.3 {
                DynamicsType::ResonatingPeriodic
            } else if rate > -0.75 {
                DynamicsType::Chaotic
            } else {
                DynamicsType::NonResonatingPeriodic
            }
        }
        None => DynamicsType::Indeterminate,
    };

    Ok(HtaConvergenceResult {
        times,
        hta_magnitudes,
        convergence_rate,
        dynamics_type,
        omega,
        period: 1.0 / omega,
    })
}

/// Phase space classification labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseSpaceClass {
    /// |HTA| ≥ 0.01: Resonating periodic set.
    Resonating = 1,
    /// 0.0001 ≤ |HTA| < 0.01: Chaotic region.
    Chaotic = 2,
    /// |HTA| < 0.0001: Non-resonating periodic set.
    NonResonating = 3,
}

/// Classify phase space points by HTA magnitude thresholds.
///
/// Default thresholds: resonating ≥ 0.01, chaotic ≥ 0.0001.
pub fn classify_phase_space(
    hta_magnitudes: &[f64],
    resonating_threshold: f64,
    chaotic_threshold: f64,
) -> Vec<PhaseSpaceClass> {
    hta_magnitudes
        .iter()
        .map(|&m| {
            if m >= resonating_threshold {
                PhaseSpaceClass::Resonating
            } else if m >= chaotic_threshold {
                PhaseSpaceClass::Chaotic
            } else {
                PhaseSpaceClass::NonResonating
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maps::StandardMap;

    fn assert_near(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

    #[test]
    fn test_observable_eval() {
        let state = vec![0.5, 0.3];
        assert_near(Observable::Identity.eval(&state), 0.5, 1e-12);
        assert_near(Observable::SinPi.eval(&state), (PI * 0.5).sin(), 1e-12);
        assert_near(Observable::CosPi.eval(&state), (PI * 0.5).cos(), 1e-12);
        assert_near(
            Observable::SinPiXY.eval(&state),
            (PI * 0.5).sin() * (PI * 0.3).sin(),
            1e-12,
        );
    }

    #[test]
    fn test_hta_zero_frequency() {
        // omega=0: HTA reduces to time average
        let map = StandardMap { epsilon: 0.0 };
        let result =
            harmonic_time_average(&[0.5, 0.3], &map, &Observable::Identity, 1e-10, 100).unwrap();
        // With epsilon=0, x advances by y=0.3 each step (mod 1)
        // Time average of x should be ~0.5 (uniform coverage)
        assert!(result.magnitude > 0.0);
    }

    #[test]
    fn test_hta_basic() {
        let map = StandardMap { epsilon: 0.12 };
        let result = harmonic_time_average(
            &[0.5, 0.3],
            &map,
            &Observable::SinPi,
            0.5, // period-2 detection
            10000,
        )
        .unwrap();

        assert!(result.magnitude >= 0.0);
        assert_near(result.omega, 0.5, 1e-12);
        assert_near(result.period, 2.0, 1e-12);
        assert_eq!(result.n_iter, 10000);
    }

    #[test]
    fn test_hta_from_values() {
        // Pure cosine at frequency omega should give large |HTA|
        let omega = 0.1;
        let n = 10000;
        let values: Vec<f64> = (0..n)
            .map(|k| (2.0 * PI * k as f64 * omega).cos())
            .collect();
        let hta = hta_from_values(&values, omega);
        // Should be ~0.5 (half the amplitude due to complex exponential)
        assert!(hta.norm() > 0.4);
    }

    #[test]
    fn test_hta_convergence_analysis() {
        let map = StandardMap { epsilon: 0.12 };
        let result = hta_convergence(
            &[0.5, 0.25], // periodic region
            &map,
            &Observable::SinPi,
            0.5,
            50000,
            None,
        )
        .unwrap();

        assert!(!result.times.is_empty());
        assert_eq!(result.hta_magnitudes.len(), result.times.len());
        assert!(result.convergence_rate.is_some());
    }

    #[test]
    fn test_classify_phase_space() {
        let mags = vec![0.15, 0.001, 1e-5, 0.08, 0.0005];
        let classes = classify_phase_space(&mags, 0.01, 1e-4);
        assert_eq!(classes[0], PhaseSpaceClass::Resonating);
        assert_eq!(classes[1], PhaseSpaceClass::Chaotic);
        assert_eq!(classes[2], PhaseSpaceClass::NonResonating);
        assert_eq!(classes[3], PhaseSpaceClass::Resonating);
        assert_eq!(classes[4], PhaseSpaceClass::Chaotic);
    }

    #[test]
    fn test_hta_error_zero_iter() {
        let map = StandardMap::default();
        assert!(harmonic_time_average(&[0.5, 0.3], &map, &Observable::SinPi, 0.5, 0).is_err());
    }
}
