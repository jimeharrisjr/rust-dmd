use extendr_api::prelude::*;
use koopman_dmd as kdmd;

// ============================================================================
// Helper conversions
// ============================================================================

fn rmatrix_to_faer(x: RMatrix<f64>) -> faer::Mat<f64> {
    let nrows = x.nrows();
    let ncols = x.ncols();
    let data = x.as_real_slice().unwrap();
    let mut m = faer::Mat::<f64>::zeros(nrows, ncols);
    // R matrices are column-major
    for j in 0..ncols {
        for i in 0..nrows {
            m[(i, j)] = data[j * nrows + i];
        }
    }
    m
}

fn faer_to_rmatrix(m: &faer::Mat<f64>) -> RMatrix<f64> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    // R matrices are column-major
    let mut data = vec![0.0; nrows * ncols];
    for j in 0..ncols {
        for i in 0..nrows {
            data[j * nrows + i] = m[(i, j)];
        }
    }
    RMatrix::new_matrix(nrows, ncols, |r, c| data[c * nrows + r])
}

// ============================================================================
// Core DMD
// ============================================================================

/// Perform Dynamic Mode Decomposition.
///
/// @param x Numeric matrix (n_vars x n_time).
/// @param rank Integer truncation rank, or NULL for automatic.
/// @param center Logical, center data by subtracting row means.
/// @param dt Numeric time step.
/// @param lifting Character lifting type or NULL.
/// @param lifting_param Integer lifting parameter or NULL.
/// @return A list containing the DMD decomposition.
/// @export
#[extendr]
fn rust_dmd(
    x: RMatrix<f64>,
    rank: Nullable<i32>,
    center: bool,
    dt: f64,
    lifting: Nullable<String>,
    lifting_param: Nullable<i32>,
) -> Result<List> {
    let mat = rmatrix_to_faer(x);

    let lifting_config = match lifting {
        Nullable::NotNull(ref s) => match s.as_str() {
            "polynomial" | "poly2" | "poly3" | "poly4" => {
                let degree = match &lifting_param {
                    Nullable::NotNull(p) => *p as usize,
                    Nullable::Null => match s.as_str() {
                        "poly3" => 3,
                        "poly4" => 4,
                        _ => 2,
                    },
                };
                Some(kdmd::LiftingConfig::Polynomial { degree })
            }
            "polynomial_cross" | "poly_cross2" | "poly_cross3" => {
                let degree = match &lifting_param {
                    Nullable::NotNull(p) => *p as usize,
                    Nullable::Null => {
                        if s.contains('3') {
                            3
                        } else {
                            2
                        }
                    }
                };
                Some(kdmd::LiftingConfig::PolynomialCross { degree })
            }
            "trigonometric" | "trig" | "trig2" => {
                let harmonics = match &lifting_param {
                    Nullable::NotNull(p) => *p as usize,
                    Nullable::Null => {
                        if s.contains('2') {
                            2
                        } else {
                            1
                        }
                    }
                };
                Some(kdmd::LiftingConfig::Trigonometric { harmonics })
            }
            "delay" | "delay2" | "delay3" | "delay5" => {
                let delays = match &lifting_param {
                    Nullable::NotNull(p) => *p as usize,
                    Nullable::Null => match s.as_str() {
                        "delay2" => 2,
                        "delay3" => 3,
                        "delay5" => 5,
                        _ => 5,
                    },
                };
                Some(kdmd::LiftingConfig::Delay { delays })
            }
            _ => {
                return Err(Error::Other(format!(
                    "Unknown lifting type: '{}'. Use 'polynomial', 'polynomial_cross', 'trigonometric', or 'delay'.",
                    s
                )));
            }
        },
        Nullable::Null => None,
    };

    let config = kdmd::DmdConfig {
        rank: match rank {
            Nullable::NotNull(r) => Some(r as usize),
            Nullable::Null => None,
        },
        center,
        dt,
        lifting: lifting_config,
    };

    let result = kdmd::dmd(&mat, &config).map_err(|e| Error::Other(e.to_string()))?;

    // Convert eigenvalues to R complex: two vectors (re, im)
    let eig_re: Vec<f64> = result.eigenvalues.iter().map(|e| e.re).collect();
    let eig_im: Vec<f64> = result.eigenvalues.iter().map(|e| e.im).collect();

    // Modes: (n_vars, rank) with separate re/im matrices
    let n_vars = result.modes.len();
    let r = result.rank;
    let mut modes_re = vec![0.0; n_vars * r];
    let mut modes_im = vec![0.0; n_vars * r];
    for j in 0..r {
        for i in 0..n_vars {
            modes_re[j * n_vars + i] = result.modes[i][j].re;
            modes_im[j * n_vars + i] = result.modes[i][j].im;
        }
    }

    let amp_re: Vec<f64> = result.amplitudes.iter().map(|a| a.re).collect();
    let amp_im: Vec<f64> = result.amplitudes.iter().map(|a| a.im).collect();

    let sv: Vec<f64> = result.svd.s.clone();

    // Store the original matrix for later error/residual computations
    let x_first: Vec<f64> = (0..mat.nrows()).map(|i| mat[(i, 0)]).collect();

    Ok(list!(
        rank = result.rank as i32,
        data_dim = vec![result.data_dim.0 as i32, result.data_dim.1 as i32],
        center = result.center,
        dt = result.dt,
        eigenvalues_re = eig_re,
        eigenvalues_im = eig_im,
        modes_re = RMatrix::new_matrix(n_vars, r, |i, j| modes_re[j * n_vars + i]),
        modes_im = RMatrix::new_matrix(n_vars, r, |i, j| modes_im[j * n_vars + i]),
        amplitudes_re = amp_re,
        amplitudes_im = amp_im,
        singular_values = sv,
        x_first = x_first,
        n_vars = n_vars as i32,
        // Serialize the DmdResult as opaque bytes for later use
        _result_ptr = result_to_raw(&result),
        _x_ptr = mat_to_raw(&mat)
    ))
}

// Store DmdResult as serialized bytes in an R raw vector.
// We use a simple approach: Box::into_raw + pointer as i64.
// This is safe because the R side will never modify the raw bytes.
fn result_to_raw(result: &kdmd::DmdResult) -> Robj {
    let boxed = Box::new(result.clone());
    let ptr = Box::into_raw(boxed) as usize as f64;
    r!(ptr)
}

fn mat_to_raw(mat: &faer::Mat<f64>) -> Robj {
    let boxed = Box::new(mat.clone());
    let ptr = Box::into_raw(boxed) as usize as f64;
    r!(ptr)
}

fn raw_to_result(ptr_val: f64) -> &'static kdmd::DmdResult {
    let ptr = ptr_val as usize as *const kdmd::DmdResult;
    unsafe { &*ptr }
}

fn raw_to_mat(ptr_val: f64) -> &'static faer::Mat<f64> {
    let ptr = ptr_val as usize as *const faer::Mat<f64>;
    unsafe { &*ptr }
}

/// Predict future states from DMD result.
/// @param result_ptr Opaque pointer to DmdResult.
/// @param n_ahead Number of steps to predict.
/// @param x0 Optional initial condition vector.
/// @param method "modes" or "matrix".
/// @return Numeric matrix of predictions.
#[extendr]
fn rust_dmd_predict(
    result_ptr: f64,
    n_ahead: i32,
    x0: Nullable<Vec<f64>>,
    method: &str,
) -> Result<RMatrix<f64>> {
    let result = raw_to_result(result_ptr);
    let x0_ref = match &x0 {
        Nullable::NotNull(v) => Some(v.as_slice()),
        Nullable::Null => None,
    };
    let pred = match method {
        "modes" => kdmd::predict_modes(result, n_ahead as usize, x0_ref),
        "matrix" => kdmd::predict_matrix(result, n_ahead as usize, x0_ref),
        _ => return Err(Error::Other("method must be 'modes' or 'matrix'".into())),
    }
    .map_err(|e| Error::Other(e.to_string()))?;
    Ok(faer_to_rmatrix(&pred))
}

/// Reconstruct data from DMD modes.
/// @param result_ptr Opaque pointer to DmdResult.
/// @param n_steps Number of time steps.
/// @param modes_subset Optional integer vector of mode indices (1-based).
/// @return Numeric matrix.
#[extendr]
fn rust_dmd_reconstruct(
    result_ptr: f64,
    n_steps: i32,
    modes_subset: Nullable<Vec<i32>>,
) -> Result<RMatrix<f64>> {
    let result = raw_to_result(result_ptr);
    let subset: Option<Vec<usize>> = match modes_subset {
        Nullable::NotNull(v) => Some(v.iter().map(|&i| (i - 1) as usize).collect()),
        Nullable::Null => None,
    };
    let recon = kdmd::dmd_reconstruct(result, n_steps as usize, subset.as_deref())
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(faer_to_rmatrix(&recon))
}

/// Analyze eigenvalue spectrum.
/// @param result_ptr Opaque pointer.
/// @param dt Time step.
/// @return List of mode information.
#[extendr]
fn rust_dmd_spectrum(result_ptr: f64, dt: f64) -> List {
    let result = raw_to_result(result_ptr);
    let spec = kdmd::dmd_spectrum(result, dt);

    let indices: Vec<i32> = spec.iter().map(|m| m.index as i32).collect();
    let magnitudes: Vec<f64> = spec.iter().map(|m| m.magnitude).collect();
    let phases: Vec<f64> = spec.iter().map(|m| m.phase).collect();
    let frequencies: Vec<f64> = spec.iter().map(|m| m.frequency).collect();
    let periods: Vec<f64> = spec.iter().map(|m| m.period).collect();
    let growth_rates: Vec<f64> = spec.iter().map(|m| m.growth_rate).collect();
    let amplitudes: Vec<f64> = spec.iter().map(|m| m.amplitude).collect();
    let stabilities: Vec<String> = spec.iter().map(|m| m.stability.to_string()).collect();

    list!(
        index = indices,
        magnitude = magnitudes,
        phase = phases,
        frequency = frequencies,
        period = periods,
        growth_rate = growth_rates,
        amplitude = amplitudes,
        stability = stabilities
    )
}

/// Analyze system stability.
/// @param result_ptr Opaque pointer.
/// @param tol Tolerance for marginal classification.
/// @return List with stability info.
#[extendr]
fn rust_dmd_stability(result_ptr: f64, tol: f64) -> List {
    let result = raw_to_result(result_ptr);
    let stab = kdmd::dmd_stability(result, tol);
    list!(
        is_stable = stab.is_stable,
        is_unstable = stab.is_unstable,
        is_marginal = stab.is_marginal,
        spectral_radius = stab.spectral_radius
    )
}

/// Compute reconstruction error.
/// @param result_ptr Opaque pointer to DmdResult.
/// @param x_ptr Opaque pointer to original data.
/// @return List with error metrics.
#[extendr]
fn rust_dmd_error(result_ptr: f64, x_ptr: f64) -> Result<List> {
    let result = raw_to_result(result_ptr);
    let x = raw_to_mat(x_ptr);
    let err = kdmd::dmd_error(result, x).map_err(|e| Error::Other(e.to_string()))?;
    Ok(list!(
        rmse = err.rmse,
        mae = err.mae,
        mape = err.mape,
        relative_error = err.relative_error
    ))
}

/// Get dominant mode indices.
/// @param result_ptr Opaque pointer.
/// @param n Number of modes.
/// @param criterion "amplitude", "energy", or "stability".
/// @return Integer vector of 1-based indices.
#[extendr]
fn rust_dmd_dominant_modes(result_ptr: f64, n: i32, criterion: &str) -> Result<Vec<i32>> {
    let result = raw_to_result(result_ptr);
    let crit = match criterion {
        "amplitude" => kdmd::DominantCriterion::Amplitude,
        "energy" => kdmd::DominantCriterion::Energy,
        "stability" => kdmd::DominantCriterion::Stability,
        _ => {
            return Err(Error::Other(
                "criterion must be 'amplitude', 'energy', or 'stability'".into(),
            ))
        }
    };
    let indices = kdmd::dmd_dominant_modes(result, n as usize, crit);
    Ok(indices.iter().map(|&i| (i + 1) as i32).collect()) // 1-based
}

/// Compute residual analysis.
/// @param result_ptr Opaque pointer.
/// @param x_ptr Opaque pointer to original data.
/// @return List with residual metrics.
#[extendr]
fn rust_dmd_residual(result_ptr: f64, x_ptr: f64) -> Result<List> {
    let result = raw_to_result(result_ptr);
    let x = raw_to_mat(x_ptr);
    let res = kdmd::dmd_residual(result, x).map_err(|e| Error::Other(e.to_string()))?;
    Ok(list!(
        residual_norm = res.residual_norm,
        residual_relative = res.residual_relative
    ))
}

// ============================================================================
// Hankel-DMD
// ============================================================================

/// Perform Hankel-DMD.
/// @param y Numeric matrix (n_obs x n_time).
/// @param delays Integer or NULL.
/// @param rank Integer or NULL.
/// @param dt Numeric time step.
/// @return List containing Hankel-DMD result.
#[extendr]
fn rust_hankel_dmd(
    y: RMatrix<f64>,
    delays: Nullable<i32>,
    rank: Nullable<i32>,
    dt: f64,
) -> Result<List> {
    let mat = rmatrix_to_faer(y);
    let config = kdmd::HankelConfig {
        delays: match delays {
            Nullable::NotNull(d) => Some(d as usize),
            Nullable::Null => None,
        },
        rank: match rank {
            Nullable::NotNull(r) => Some(r as usize),
            Nullable::Null => None,
        },
        dt,
    };
    let result = kdmd::hankel_dmd(&mat, &config).map_err(|e| Error::Other(e.to_string()))?;

    let eig_re: Vec<f64> = result.eigenvalues.iter().map(|e| e.re).collect();
    let eig_im: Vec<f64> = result.eigenvalues.iter().map(|e| e.im).collect();

    let boxed = Box::new(result.clone());
    let ptr = Box::into_raw(boxed) as usize as f64;

    Ok(list!(
        rank = result.rank as i32,
        delays = result.delays as i32,
        n_obs = result.n_obs as i32,
        residual = result.residual,
        eigenvalues_re = eig_re,
        eigenvalues_im = eig_im,
        _result_ptr = ptr
    ))
}

/// Reconstruct from Hankel-DMD.
#[extendr]
fn rust_hankel_reconstruct(result_ptr: f64, n_steps: i32) -> Result<RMatrix<f64>> {
    let ptr = result_ptr as usize as *const kdmd::HankelDmdResult;
    let result = unsafe { &*ptr };
    let recon = kdmd::hankel_reconstruct(result, n_steps as usize)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(faer_to_rmatrix(&recon))
}

/// Predict from Hankel-DMD.
#[extendr]
fn rust_hankel_predict(result_ptr: f64, n_ahead: i32) -> Result<RMatrix<f64>> {
    let ptr = result_ptr as usize as *const kdmd::HankelDmdResult;
    let result = unsafe { &*ptr };
    let pred =
        kdmd::hankel_predict(result, n_ahead as usize).map_err(|e| Error::Other(e.to_string()))?;
    Ok(faer_to_rmatrix(&pred))
}

// ============================================================================
// GLA
// ============================================================================

/// Perform Generalized Laplace Analysis.
/// @param y Numeric matrix (n_obs x n_time).
/// @param eigenvalues_re Optional real parts of known eigenvalues.
/// @param eigenvalues_im Optional imaginary parts of known eigenvalues.
/// @param n_eigenvalues Number of eigenvalues to estimate.
/// @param tol Convergence tolerance.
/// @param max_iter Maximum iterations or NULL.
/// @return List containing GLA result.
#[extendr]
fn rust_gla(
    y: RMatrix<f64>,
    eigenvalues_re: Nullable<Vec<f64>>,
    eigenvalues_im: Nullable<Vec<f64>>,
    n_eigenvalues: i32,
    tol: f64,
    max_iter: Nullable<i32>,
) -> Result<List> {
    let mat = rmatrix_to_faer(y);

    let evs = match (&eigenvalues_re, &eigenvalues_im) {
        (Nullable::NotNull(re), Nullable::NotNull(im)) => Some(
            re.iter()
                .zip(im.iter())
                .map(|(&r, &i)| kdmd::C64::new(r, i))
                .collect(),
        ),
        _ => None,
    };

    let config = kdmd::GlaConfig {
        eigenvalues: evs,
        n_eigenvalues: n_eigenvalues as usize,
        tol,
        max_iter: match max_iter {
            Nullable::NotNull(m) => Some(m as usize),
            Nullable::Null => None,
        },
    };

    let result = kdmd::gla(&mat, &config).map_err(|e| Error::Other(e.to_string()))?;

    let eig_re: Vec<f64> = result.eigenvalues.iter().map(|e| e.re).collect();
    let eig_im: Vec<f64> = result.eigenvalues.iter().map(|e| e.im).collect();
    let convergence: Vec<bool> = result.convergence.clone();
    let residuals: Vec<f64> = result.residuals.clone();

    let boxed = Box::new(result.clone());
    let ptr = Box::into_raw(boxed) as usize as f64;

    Ok(list!(
        n_obs = result.n_obs as i32,
        n_time = result.n_time as i32,
        eigenvalues_re = eig_re,
        eigenvalues_im = eig_im,
        convergence = convergence,
        residuals = residuals,
        _result_ptr = ptr
    ))
}

/// Predict from GLA.
#[extendr]
fn rust_gla_predict(result_ptr: f64, n_ahead: i32) -> Result<RMatrix<f64>> {
    let ptr = result_ptr as usize as *const kdmd::GlaResult;
    let result = unsafe { &*ptr };
    let pred =
        kdmd::gla_predict(result, n_ahead as usize).map_err(|e| Error::Other(e.to_string()))?;
    Ok(faer_to_rmatrix(&pred))
}

/// Reconstruct from GLA.
#[extendr]
fn rust_gla_reconstruct(result_ptr: f64, modes_to_use: Nullable<Vec<i32>>) -> Result<RMatrix<f64>> {
    let ptr = result_ptr as usize as *const kdmd::GlaResult;
    let result = unsafe { &*ptr };
    let subset: Option<Vec<usize>> = match modes_to_use {
        Nullable::NotNull(v) => Some(v.iter().map(|&i| (i - 1) as usize).collect()),
        Nullable::Null => None,
    };
    let recon = kdmd::gla_reconstruct(result, subset.as_deref())
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(faer_to_rmatrix(&recon))
}

// ============================================================================
// Maps
// ============================================================================

fn make_map(map_name: &str, params: &List) -> std::result::Result<Box<dyn kdmd::MapFn>, Error> {
    let get_f64 = |key: &str, default: f64| -> f64 {
        params
            .dollar(key)
            .ok()
            .and_then(|v: Robj| v.as_real())
            .unwrap_or(default)
    };

    match map_name {
        "standard" => Ok(Box::new(kdmd::StandardMap {
            epsilon: get_f64("epsilon", 0.12),
        })),
        "froeschle" => Ok(Box::new(kdmd::FroeschleMap {
            epsilon1: get_f64("epsilon1", 0.02),
            epsilon2: get_f64("epsilon2", 0.02),
            eta: get_f64("eta", 0.01),
        })),
        "extended_standard" => Ok(Box::new(kdmd::ExtendedStandardMap {
            epsilon: get_f64("epsilon", 0.01),
            delta: get_f64("delta", 0.001),
        })),
        "henon" => Ok(Box::new(kdmd::HenonMap {
            a: get_f64("a", 1.4),
            b: get_f64("b", 0.3),
        })),
        "logistic" => Ok(Box::new(kdmd::LogisticMap {
            r: get_f64("r", 3.9),
        })),
        _ => Err(Error::Other(format!(
            "Unknown map: '{}'. Use 'standard', 'froeschle', 'extended_standard', 'henon', or 'logistic'.",
            map_name
        ))),
    }
}

fn parse_observable(name: &str) -> std::result::Result<kdmd::Observable, Error> {
    match name {
        "identity" => Ok(kdmd::Observable::Identity),
        "sin_pi" => Ok(kdmd::Observable::SinPi),
        "cos_pi" => Ok(kdmd::Observable::CosPi),
        "sin_pi_xy" => Ok(kdmd::Observable::SinPiXY),
        "cos_pi_xy" => Ok(kdmd::Observable::CosPiXY),
        "sin_2pi" => Ok(kdmd::Observable::Sin2Pi),
        "cos_2pi" => Ok(kdmd::Observable::Cos2Pi),
        "trig_product" => Ok(kdmd::Observable::TrigProduct),
        _ => Err(Error::Other(format!(
            "Unknown observable: '{}'. Use 'identity', 'sin_pi', 'cos_pi', etc.",
            name
        ))),
    }
}

/// Generate a trajectory from a built-in map.
/// @param map_name Character map name.
/// @param initial_condition Numeric vector.
/// @param n_iter Number of iterations.
/// @param params Named list of map parameters.
/// @return Numeric matrix (n_dim x n_iter+1).
#[extendr]
fn rust_generate_trajectory(
    map_name: &str,
    initial_condition: Vec<f64>,
    n_iter: i32,
    params: List,
) -> Result<RMatrix<f64>> {
    let map = make_map(map_name, &params)?;
    let traj = kdmd::generate_trajectory(&initial_condition, map.as_ref(), n_iter as usize);
    Ok(faer_to_rmatrix(&traj))
}

// ============================================================================
// Harmonic Time Average
// ============================================================================

/// Compute harmonic time average.
/// @param map_name Character map name.
/// @param initial_condition Numeric vector.
/// @param observable Character observable name.
/// @param omega Numeric frequency.
/// @param n_iter Integer iterations.
/// @param params Named list of map parameters.
/// @return List with magnitude, phase, hta_re, hta_im.
#[extendr]
fn rust_harmonic_time_average(
    map_name: &str,
    initial_condition: Vec<f64>,
    observable: &str,
    omega: f64,
    n_iter: i32,
    params: List,
) -> Result<List> {
    let obs = parse_observable(observable)?;
    let map = make_map(map_name, &params)?;
    let result = kdmd::harmonic_time_average(
        &initial_condition,
        map.as_ref(),
        &obs,
        omega,
        n_iter as usize,
    )
    .map_err(|e| Error::Other(e.to_string()))?;

    Ok(list!(
        magnitude = result.magnitude,
        phase = result.phase,
        hta_re = result.hta.re,
        hta_im = result.hta.im
    ))
}

/// Compute mesochronic harmonic plot.
/// @param map_name Character map name.
/// @param x_range Numeric vector c(min, max).
/// @param y_range Numeric vector c(min, max).
/// @param resolution Integer grid resolution.
/// @param observable Character observable name.
/// @param omega Numeric frequency.
/// @param n_iter Integer iterations.
/// @param params Named list of map parameters.
/// @return List with hta_matrix, phase_matrix, x_coords, y_coords.
#[extendr]
fn rust_mesochronic_compute(
    map_name: &str,
    x_range: Vec<f64>,
    y_range: Vec<f64>,
    resolution: i32,
    observable: &str,
    omega: f64,
    n_iter: i32,
    params: List,
) -> Result<List> {
    let obs = parse_observable(observable)?;
    let map = make_map(map_name, &params)?;
    let result = kdmd::mesochronic_compute(
        map.as_ref(),
        (x_range[0], x_range[1]),
        (y_range[0], y_range[1]),
        resolution as usize,
        &obs,
        omega,
        n_iter as usize,
    )
    .map_err(|e| Error::Other(e.to_string()))?;

    let res = resolution as usize;
    // Flatten 2D vectors to column-major matrices
    let mut hta_data = vec![0.0; res * res];
    let mut phase_data = vec![0.0; res * res];
    for j in 0..res {
        for i in 0..res {
            hta_data[j * res + i] = result.hta_matrix[i][j];
            phase_data[j * res + i] = result.phase_matrix[i][j];
        }
    }

    Ok(list!(
        hta_matrix = RMatrix::new_matrix(res, res, |i, j| hta_data[j * res + i]),
        phase_matrix = RMatrix::new_matrix(res, res, |i, j| phase_data[j * res + i]),
        x_coords = result.x_coords,
        y_coords = result.y_coords
    ))
}

/// Classify phase space points.
/// @param hta_magnitudes Numeric vector of |HTA| values.
/// @param resonating_threshold Numeric threshold.
/// @param chaotic_threshold Numeric threshold.
/// @return Integer vector (1=resonating, 2=chaotic, 3=non-resonating).
#[extendr]
fn rust_classify_phase_space(
    hta_magnitudes: Vec<f64>,
    resonating_threshold: f64,
    chaotic_threshold: f64,
) -> Vec<i32> {
    let classes =
        kdmd::classify_phase_space(&hta_magnitudes, resonating_threshold, chaotic_threshold);
    classes.iter().map(|c| *c as i32).collect()
}

/// Analyze HTA convergence.
/// @param map_name Character map name.
/// @param initial_condition Numeric vector.
/// @param observable Character observable name.
/// @param omega Numeric frequency.
/// @param n_iter Integer iterations.
/// @param params Named list of map parameters.
/// @return List with times, hta_magnitudes, convergence_rate, dynamics_type.
#[extendr]
fn rust_hta_convergence(
    map_name: &str,
    initial_condition: Vec<f64>,
    observable: &str,
    omega: f64,
    n_iter: i32,
    params: List,
) -> Result<List> {
    let obs = parse_observable(observable)?;
    let map = make_map(map_name, &params)?;
    let result = kdmd::hta_convergence(
        &initial_condition,
        map.as_ref(),
        &obs,
        omega,
        n_iter as usize,
        None,
    )
    .map_err(|e| Error::Other(e.to_string()))?;

    // Convert usize times to f64 for R
    let times: Vec<f64> = result.times.iter().map(|&t| t as f64).collect();

    Ok(list!(
        times = times,
        hta_magnitudes = result.hta_magnitudes,
        convergence_rate = match result.convergence_rate {
            Some(r) => Nullable::NotNull(r),
            None => Nullable::Null,
        },
        dynamics_type = result.dynamics_type.to_string()
    ))
}

// ============================================================================
// Module registration
// ============================================================================

extendr_module! {
    mod koopman_dmd_r;
    fn rust_dmd;
    fn rust_dmd_predict;
    fn rust_dmd_reconstruct;
    fn rust_dmd_spectrum;
    fn rust_dmd_stability;
    fn rust_dmd_error;
    fn rust_dmd_dominant_modes;
    fn rust_dmd_residual;
    fn rust_hankel_dmd;
    fn rust_hankel_reconstruct;
    fn rust_hankel_predict;
    fn rust_gla;
    fn rust_gla_predict;
    fn rust_gla_reconstruct;
    fn rust_generate_trajectory;
    fn rust_harmonic_time_average;
    fn rust_mesochronic_compute;
    fn rust_classify_phase_space;
    fn rust_hta_convergence;
}
