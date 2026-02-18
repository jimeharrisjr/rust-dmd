use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::koopman_dmd as kdmd;

// ============================================================================
// Helper conversions
// ============================================================================

fn mat_to_faer(arr: &Array2<f64>) -> faer::Mat<f64> {
    let (nrows, ncols) = arr.dim();
    let mut m = faer::Mat::<f64>::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            m[(i, j)] = arr[[i, j]];
        }
    }
    m
}

fn faer_to_array2(m: &faer::Mat<f64>) -> Array2<f64> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

fn dmd_err_to_py(e: kdmd::DmdError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ============================================================================
// DMD class
// ============================================================================

/// Dynamic Mode Decomposition result.
///
/// Compute DMD by calling `DMD(X, ...)` where X is a 2D numpy array
/// with shape (n_variables, n_time_steps).
#[pyclass]
#[derive(Clone)]
struct DMD {
    result: kdmd::DmdResult,
    x_original: faer::Mat<f64>,
}

#[pymethods]
impl DMD {
    /// Create a DMD decomposition.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray
    ///     Data matrix (n_vars x n_time).
    /// rank : int, optional
    ///     Truncation rank. None for automatic.
    /// center : bool
    ///     Center data by subtracting row means.
    /// dt : float
    ///     Time step between snapshots.
    /// lifting : str, optional
    ///     Lifting type: "polynomial", "trigonometric", "delay".
    /// lifting_param : int, optional
    ///     Lifting parameter (degree, harmonics, or delays).
    #[new]
    #[pyo3(signature = (x, rank=None, center=false, dt=1.0, lifting=None, lifting_param=None))]
    fn new(
        x: PyReadonlyArray2<f64>,
        rank: Option<usize>,
        center: bool,
        dt: f64,
        lifting: Option<&str>,
        lifting_param: Option<usize>,
    ) -> PyResult<Self> {
        let arr = x.as_array().to_owned();
        let mat = mat_to_faer(&arr);

        let lifting_config = match lifting {
            Some("polynomial") => Some(kdmd::LiftingConfig::Polynomial {
                degree: lifting_param.unwrap_or(2),
            }),
            Some("polynomial_cross") => Some(kdmd::LiftingConfig::PolynomialCross {
                degree: lifting_param.unwrap_or(2),
            }),
            Some("trigonometric") => Some(kdmd::LiftingConfig::Trigonometric {
                harmonics: lifting_param.unwrap_or(2),
            }),
            Some("delay") => Some(kdmd::LiftingConfig::Delay {
                delays: lifting_param.unwrap_or(5),
            }),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unknown lifting type: '{other}'. Use 'polynomial', 'polynomial_cross', 'trigonometric', or 'delay'"
                )))
            }
            None => None,
        };

        let config = kdmd::DmdConfig {
            rank,
            center,
            dt,
            lifting: lifting_config,
        };

        let result = kdmd::dmd(&mat, &config).map_err(dmd_err_to_py)?;
        Ok(DMD {
            result,
            x_original: mat,
        })
    }

    /// Truncation rank used.
    #[getter]
    fn rank(&self) -> usize {
        self.result.rank
    }

    /// Data dimensions (n_vars, n_time).
    #[getter]
    fn data_dim(&self) -> (usize, usize) {
        self.result.data_dim
    }

    /// Whether centering was applied.
    #[getter]
    fn center(&self) -> bool {
        self.result.center
    }

    /// Time step.
    #[getter]
    fn dt(&self) -> f64 {
        self.result.dt
    }

    /// Eigenvalues as complex pairs (re, im).
    #[getter]
    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.result.eigenvalues.len();
        let arr = Array2::from_shape_fn((n, 2), |(i, _j)| {
            if _j == 0 {
                self.result.eigenvalues[i].re
            } else {
                self.result.eigenvalues[i].im
            }
        });
        arr.into_pyarray(py)
    }

    /// DMD modes as complex array (n_vars x rank x 2) where last dim is [re, im].
    #[getter]
    fn modes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n_vars = self.result.modes.len();
        let rank = self.result.rank;
        // Return as (n_vars, rank*2) where columns alternate re, im
        let arr = Array2::from_shape_fn((n_vars, rank * 2), |(i, j)| {
            let mode_idx = j / 2;
            if j % 2 == 0 {
                self.result.modes[i][mode_idx].re
            } else {
                self.result.modes[i][mode_idx].im
            }
        });
        arr.into_pyarray(py)
    }

    /// Amplitudes as complex pairs (rank x 2) where columns are [re, im].
    #[getter]
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.result.amplitudes.len();
        let arr = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                self.result.amplitudes[i].re
            } else {
                self.result.amplitudes[i].im
            }
        });
        arr.into_pyarray(py)
    }

    /// Singular values from the truncated SVD.
    #[getter]
    fn singular_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.result.svd.s.clone()).into_pyarray(py)
    }

    /// Predict future states.
    ///
    /// Parameters
    /// ----------
    /// n_ahead : int
    ///     Number of time steps to predict.
    /// x0 : numpy.ndarray, optional
    ///     Initial condition. If None, uses stored first snapshot.
    /// method : str
    ///     "modes" (default) or "matrix".
    #[pyo3(signature = (n_ahead, x0=None, method="modes"))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        n_ahead: usize,
        x0: Option<PyReadonlyArray1<f64>>,
        method: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x0_slice: Option<Vec<f64>> = x0.map(|a| a.as_array().to_vec());
        let x0_ref = x0_slice.as_deref();

        let pred = match method {
            "modes" => kdmd::predict_modes(&self.result, n_ahead, x0_ref),
            "matrix" => kdmd::predict_matrix(&self.result, n_ahead, x0_ref),
            _ => return Err(PyValueError::new_err("method must be 'modes' or 'matrix'")),
        }
        .map_err(dmd_err_to_py)?;

        Ok(faer_to_array2(&pred).into_pyarray(py))
    }

    /// Reconstruct data from DMD modes.
    ///
    /// Parameters
    /// ----------
    /// n_steps : int
    ///     Number of time steps to reconstruct.
    /// modes_subset : list of int, optional
    ///     Indices of modes to use. None for all.
    #[pyo3(signature = (n_steps, modes_subset=None))]
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        n_steps: usize,
        modes_subset: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let subset = modes_subset.as_deref();
        let recon = kdmd::dmd_reconstruct(&self.result, n_steps, subset).map_err(dmd_err_to_py)?;
        Ok(faer_to_array2(&recon).into_pyarray(py))
    }

    /// Analyze the eigenvalue spectrum.
    ///
    /// Returns a list of dicts with mode information.
    fn spectrum(&self, py: Python<'_>) -> PyResult<PyObject> {
        let spec = kdmd::dmd_spectrum(&self.result, self.result.dt);
        let list = pyo3::types::PyList::empty(py);
        for m in &spec {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("index", m.index)?;
            dict.set_item("magnitude", m.magnitude)?;
            dict.set_item("phase", m.phase)?;
            dict.set_item("frequency", m.frequency)?;
            dict.set_item("period", m.period)?;
            dict.set_item("growth_rate", m.growth_rate)?;
            dict.set_item("amplitude", m.amplitude)?;
            dict.set_item("stability", m.stability.to_string())?;
            list.append(dict)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Analyze system stability.
    fn stability(&self) -> PyResult<(bool, bool, bool, f64)> {
        let stab = kdmd::dmd_stability(&self.result, 1e-6);
        Ok((
            stab.is_stable,
            stab.is_unstable,
            stab.is_marginal,
            stab.spectral_radius,
        ))
    }

    /// Compute reconstruction error metrics.
    fn error(&self) -> PyResult<(f64, f64, f64, f64)> {
        let err = kdmd::dmd_error(&self.result, &self.x_original).map_err(dmd_err_to_py)?;
        Ok((err.rmse, err.mae, err.mape, err.relative_error))
    }

    /// Get indices of dominant modes.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Number of modes to return.
    /// criterion : str
    ///     "amplitude" (default), "energy", or "stability".
    #[pyo3(signature = (n, criterion="amplitude"))]
    fn dominant_modes(&self, n: usize, criterion: &str) -> PyResult<Vec<usize>> {
        let crit = match criterion {
            "amplitude" => kdmd::DominantCriterion::Amplitude,
            "energy" => kdmd::DominantCriterion::Energy,
            "stability" => kdmd::DominantCriterion::Stability,
            _ => {
                return Err(PyValueError::new_err(
                    "criterion must be 'amplitude', 'energy', or 'stability'",
                ))
            }
        };
        Ok(kdmd::dmd_dominant_modes(&self.result, n, crit))
    }

    /// Compute residual analysis.
    fn residual(&self) -> PyResult<(f64, f64)> {
        let res = kdmd::dmd_residual(&self.result, &self.x_original).map_err(dmd_err_to_py)?;
        Ok((res.residual_norm, res.residual_relative))
    }

    fn __repr__(&self) -> String {
        format!(
            "DMD(rank={}, data_dim=({}, {}), center={})",
            self.result.rank, self.result.data_dim.0, self.result.data_dim.1, self.result.center
        )
    }
}

// ============================================================================
// HankelDMD class
// ============================================================================

/// Hankel-DMD (time-delay embedding DMD) result.
#[pyclass]
#[derive(Clone)]
struct HankelDMD {
    result: kdmd::HankelDmdResult,
}

#[pymethods]
impl HankelDMD {
    /// Create a Hankel-DMD decomposition.
    ///
    /// Parameters
    /// ----------
    /// y : numpy.ndarray
    ///     Time series data (n_obs x n_time).
    /// delays : int, optional
    ///     Number of delays. None for automatic.
    /// rank : int, optional
    ///     Truncation rank. None for automatic.
    /// dt : float
    ///     Time step.
    #[new]
    #[pyo3(signature = (y, delays=None, rank=None, dt=1.0))]
    fn new(
        y: PyReadonlyArray2<f64>,
        delays: Option<usize>,
        rank: Option<usize>,
        dt: f64,
    ) -> PyResult<Self> {
        let arr = y.as_array().to_owned();
        let mat = mat_to_faer(&arr);
        let config = kdmd::HankelConfig { delays, rank, dt };
        let result = kdmd::hankel_dmd(&mat, &config).map_err(dmd_err_to_py)?;
        Ok(HankelDMD { result })
    }

    #[getter]
    fn rank(&self) -> usize {
        self.result.rank
    }

    #[getter]
    fn delays(&self) -> usize {
        self.result.delays
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.result.n_obs
    }

    #[getter]
    fn residual(&self) -> f64 {
        self.result.residual
    }

    #[getter]
    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.result.eigenvalues.len();
        let arr = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                self.result.eigenvalues[i].re
            } else {
                self.result.eigenvalues[i].im
            }
        });
        arr.into_pyarray(py)
    }

    /// Reconstruct the time series.
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let recon = kdmd::hankel_reconstruct(&self.result, n_steps).map_err(dmd_err_to_py)?;
        Ok(faer_to_array2(&recon).into_pyarray(py))
    }

    /// Predict future values.
    fn predict<'py>(&self, py: Python<'py>, n_ahead: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pred = kdmd::hankel_predict(&self.result, n_ahead).map_err(dmd_err_to_py)?;
        Ok(faer_to_array2(&pred).into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "HankelDMD(rank={}, delays={}, n_obs={})",
            self.result.rank, self.result.delays, self.result.n_obs
        )
    }
}

// ============================================================================
// GLA class
// ============================================================================

/// Generalized Laplace Analysis result.
#[pyclass]
#[derive(Clone)]
struct GLA {
    result: kdmd::GlaResult,
}

#[pymethods]
impl GLA {
    /// Create a GLA decomposition.
    ///
    /// Parameters
    /// ----------
    /// y : numpy.ndarray
    ///     Trajectory data (n_obs x n_time).
    /// eigenvalues : list of (float, float), optional
    ///     Known eigenvalues as (re, im) pairs. None for auto.
    /// n_eigenvalues : int
    ///     Number of eigenvalues to estimate if not provided.
    /// tol : float
    ///     Convergence tolerance.
    /// max_iter : int, optional
    ///     Maximum iterations.
    #[new]
    #[pyo3(signature = (y, eigenvalues=None, n_eigenvalues=5, tol=1e-6, max_iter=None))]
    fn new(
        y: PyReadonlyArray2<f64>,
        eigenvalues: Option<Vec<(f64, f64)>>,
        n_eigenvalues: usize,
        tol: f64,
        max_iter: Option<usize>,
    ) -> PyResult<Self> {
        let arr = y.as_array().to_owned();
        let mat = mat_to_faer(&arr);

        let evs = eigenvalues.map(|pairs| {
            pairs
                .into_iter()
                .map(|(re, im)| kdmd::C64::new(re, im))
                .collect()
        });

        let config = kdmd::GlaConfig {
            eigenvalues: evs,
            n_eigenvalues,
            tol,
            max_iter,
        };

        let result = kdmd::gla(&mat, &config).map_err(dmd_err_to_py)?;
        Ok(GLA { result })
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.result.n_obs
    }

    #[getter]
    fn n_time(&self) -> usize {
        self.result.n_time
    }

    #[getter]
    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.result.eigenvalues.len();
        let arr = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                self.result.eigenvalues[i].re
            } else {
                self.result.eigenvalues[i].im
            }
        });
        arr.into_pyarray(py)
    }

    #[getter]
    fn convergence(&self) -> Vec<bool> {
        self.result.convergence.clone()
    }

    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.result.residuals.clone()).into_pyarray(py)
    }

    /// Predict future values.
    fn predict<'py>(&self, py: Python<'py>, n_ahead: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pred = kdmd::gla_predict(&self.result, n_ahead).map_err(dmd_err_to_py)?;
        Ok(faer_to_array2(&pred).into_pyarray(py))
    }

    /// Reconstruct the signal.
    #[pyo3(signature = (modes_to_use=None))]
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        modes_to_use: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let subset = modes_to_use.as_deref();
        let recon = kdmd::gla_reconstruct(&self.result, subset).map_err(dmd_err_to_py)?;
        Ok(faer_to_array2(&recon).into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "GLA(n_eigenvalues={}, n_obs={}, n_time={})",
            self.result.eigenvalues.len(),
            self.result.n_obs,
            self.result.n_time
        )
    }
}

// ============================================================================
// Maps
// ============================================================================

/// Iterate a built-in map and return the trajectory.
///
/// Parameters
/// ----------
/// map_name : str
///     One of: "standard", "froeschle", "extended_standard", "henon", "logistic".
/// initial_condition : numpy.ndarray
///     Starting state vector.
/// n_iter : int
///     Number of iterations.
/// params : dict, optional
///     Map parameters (e.g., {"epsilon": 0.12}).
///
/// Returns
/// -------
/// numpy.ndarray
///     Trajectory matrix (n_dim x n_iter+1).
#[pyfunction]
#[pyo3(signature = (map_name, initial_condition, n_iter, params=None))]
fn generate_trajectory<'py>(
    py: Python<'py>,
    map_name: &str,
    initial_condition: PyReadonlyArray1<f64>,
    n_iter: usize,
    params: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ic = initial_condition.as_array().to_vec();

    let get_f64 = |dict: Option<&Bound<'py, pyo3::types::PyDict>>,
                   key: &str,
                   default: f64|
     -> PyResult<f64> {
        match dict {
            Some(d) => match d.get_item(key)? {
                Some(v) => v.extract::<f64>(),
                None => Ok(default),
            },
            None => Ok(default),
        }
    };

    let traj: faer::Mat<f64> = match map_name {
        "standard" => {
            let eps = get_f64(params, "epsilon", 0.12)?;
            let map = kdmd::StandardMap { epsilon: eps };
            kdmd::generate_trajectory(&ic, &map, n_iter)
        }
        "froeschle" => {
            let e1 = get_f64(params, "epsilon1", 0.02)?;
            let e2 = get_f64(params, "epsilon2", 0.02)?;
            let eta = get_f64(params, "eta", 0.01)?;
            let map = kdmd::FroeschleMap {
                epsilon1: e1,
                epsilon2: e2,
                eta,
            };
            kdmd::generate_trajectory(&ic, &map, n_iter)
        }
        "extended_standard" => {
            let eps = get_f64(params, "epsilon", 0.01)?;
            let delta = get_f64(params, "delta", 0.001)?;
            let map = kdmd::ExtendedStandardMap {
                epsilon: eps,
                delta,
            };
            kdmd::generate_trajectory(&ic, &map, n_iter)
        }
        "henon" => {
            let a = get_f64(params, "a", 1.4)?;
            let b = get_f64(params, "b", 0.3)?;
            let map = kdmd::HenonMap { a, b };
            kdmd::generate_trajectory(&ic, &map, n_iter)
        }
        "logistic" => {
            let r = get_f64(params, "r", 3.9)?;
            let map = kdmd::LogisticMap { r };
            kdmd::generate_trajectory(&ic, &map, n_iter)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown map: '{map_name}'. Use 'standard', 'froeschle', 'extended_standard', 'henon', or 'logistic'"
            )))
        }
    };

    Ok(faer_to_array2(&traj).into_pyarray(py))
}

// ============================================================================
// Harmonic Time Average
// ============================================================================

/// Compute the Harmonic Time Average for a trajectory.
///
/// Parameters
/// ----------
/// map_name : str
///     Map name (see generate_trajectory).
/// initial_condition : numpy.ndarray
///     Starting state.
/// observable : str
///     Observable name: "identity", "sin_pi", "cos_pi", "sin_pi_xy",
///     "cos_pi_xy", "sin_2pi", "cos_2pi", "trig_product".
/// omega : float
///     Frequency parameter (use 1/period).
/// n_iter : int
///     Number of iterations.
/// params : dict, optional
///     Map parameters.
///
/// Returns
/// -------
/// tuple
///     (magnitude, phase, hta_re, hta_im)
#[pyfunction]
#[pyo3(signature = (map_name, initial_condition, observable, omega, n_iter, params=None))]
fn harmonic_time_average<'py>(
    py: Python<'py>,
    map_name: &str,
    initial_condition: PyReadonlyArray1<f64>,
    observable: &str,
    omega: f64,
    n_iter: usize,
    params: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(f64, f64, f64, f64)> {
    let ic = initial_condition.as_array().to_vec();
    let obs = parse_observable(observable)?;
    let map = make_map(py, map_name, params)?;

    let result = kdmd::harmonic_time_average(&ic, map.as_ref(), &obs, omega, n_iter)
        .map_err(dmd_err_to_py)?;

    Ok((result.magnitude, result.phase, result.hta.re, result.hta.im))
}

/// Compute a Mesochronic Harmonic Plot over a 2D grid.
///
/// Parameters
/// ----------
/// map_name : str
///     Map name.
/// x_range : tuple
///     (x_min, x_max).
/// y_range : tuple
///     (y_min, y_max).
/// resolution : int
///     Grid points per dimension.
/// observable : str
///     Observable name.
/// omega : float
///     Frequency parameter.
/// n_iter : int
///     Iterations per grid point.
/// params : dict, optional
///     Map parameters.
///
/// Returns
/// -------
/// tuple
///     (hta_matrix, phase_matrix, x_coords, y_coords)
#[pyfunction]
#[pyo3(signature = (map_name, x_range, y_range, resolution, observable, omega, n_iter, params=None))]
fn mesochronic_compute<'py>(
    py: Python<'py>,
    map_name: &str,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
    observable: &str,
    omega: f64,
    n_iter: usize,
    params: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let obs = parse_observable(observable)?;
    let map = make_map(py, map_name, params)?;

    let result = kdmd::mesochronic_compute(
        map.as_ref(),
        x_range,
        y_range,
        resolution,
        &obs,
        omega,
        n_iter,
    )
    .map_err(dmd_err_to_py)?;

    let hta_arr = Array2::from_shape_fn((resolution, resolution), |(i, j)| result.hta_matrix[i][j]);
    let phase_arr =
        Array2::from_shape_fn((resolution, resolution), |(i, j)| result.phase_matrix[i][j]);
    let x_arr = Array1::from(result.x_coords);
    let y_arr = Array1::from(result.y_coords);

    Ok((
        hta_arr.into_pyarray(py),
        phase_arr.into_pyarray(py),
        x_arr.into_pyarray(py),
        y_arr.into_pyarray(py),
    ))
}

/// Classify phase space points by HTA magnitude.
///
/// Parameters
/// ----------
/// hta_magnitudes : numpy.ndarray
///     1D array of |HTA| values.
/// resonating_threshold : float
///     Threshold for resonating classification. Default 0.01.
/// chaotic_threshold : float
///     Threshold for chaotic classification. Default 0.0001.
///
/// Returns
/// -------
/// numpy.ndarray
///     1D int array: 1=resonating, 2=chaotic, 3=non-resonating.
#[pyfunction]
#[pyo3(signature = (hta_magnitudes, resonating_threshold=0.01, chaotic_threshold=0.0001))]
fn classify_phase_space<'py>(
    py: Python<'py>,
    hta_magnitudes: PyReadonlyArray1<f64>,
    resonating_threshold: f64,
    chaotic_threshold: f64,
) -> Bound<'py, PyArray1<i32>> {
    let mags = hta_magnitudes.as_array().to_vec();
    let classes = kdmd::classify_phase_space(&mags, resonating_threshold, chaotic_threshold);
    let labels: Vec<i32> = classes.iter().map(|c| *c as i32).collect();
    Array1::from(labels).into_pyarray(py)
}

/// Analyze HTA convergence for a trajectory.
///
/// Returns
/// -------
/// dict
///     {"times": [...], "hta_magnitudes": [...], "convergence_rate": float|None, "dynamics_type": str}
#[pyfunction]
#[pyo3(signature = (map_name, initial_condition, observable, omega, n_iter, params=None))]
fn hta_convergence<'py>(
    py: Python<'py>,
    map_name: &str,
    initial_condition: PyReadonlyArray1<f64>,
    observable: &str,
    omega: f64,
    n_iter: usize,
    params: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<PyObject> {
    let ic = initial_condition.as_array().to_vec();
    let obs = parse_observable(observable)?;
    let map = make_map(py, map_name, params)?;

    let result = kdmd::hta_convergence(&ic, map.as_ref(), &obs, omega, n_iter, None)
        .map_err(dmd_err_to_py)?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("times", result.times)?;
    dict.set_item("hta_magnitudes", result.hta_magnitudes)?;
    dict.set_item("convergence_rate", result.convergence_rate)?;
    dict.set_item("dynamics_type", result.dynamics_type.to_string())?;
    Ok(dict.into_any().unbind())
}

// ============================================================================
// Helper functions
// ============================================================================

fn parse_observable(name: &str) -> PyResult<kdmd::Observable> {
    match name {
        "identity" => Ok(kdmd::Observable::Identity),
        "sin_pi" => Ok(kdmd::Observable::SinPi),
        "cos_pi" => Ok(kdmd::Observable::CosPi),
        "sin_pi_xy" => Ok(kdmd::Observable::SinPiXY),
        "cos_pi_xy" => Ok(kdmd::Observable::CosPiXY),
        "sin_2pi" => Ok(kdmd::Observable::Sin2Pi),
        "cos_2pi" => Ok(kdmd::Observable::Cos2Pi),
        "trig_product" => Ok(kdmd::Observable::TrigProduct),
        _ => Err(PyValueError::new_err(format!(
            "unknown observable: '{name}'. Use 'identity', 'sin_pi', 'cos_pi', 'sin_pi_xy', 'cos_pi_xy', 'sin_2pi', 'cos_2pi', 'trig_product'"
        ))),
    }
}

fn make_map<'py>(
    _py: Python<'py>,
    map_name: &str,
    params: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Box<dyn kdmd::MapFn>> {
    let get_f64 = |key: &str, default: f64| -> PyResult<f64> {
        match params {
            Some(d) => match d.get_item(key)? {
                Some(v) => v.extract::<f64>(),
                None => Ok(default),
            },
            None => Ok(default),
        }
    };

    match map_name {
        "standard" => {
            let eps = get_f64("epsilon", 0.12)?;
            Ok(Box::new(kdmd::StandardMap { epsilon: eps }))
        }
        "froeschle" => {
            let e1 = get_f64("epsilon1", 0.02)?;
            let e2 = get_f64("epsilon2", 0.02)?;
            let eta = get_f64("eta", 0.01)?;
            Ok(Box::new(kdmd::FroeschleMap {
                epsilon1: e1,
                epsilon2: e2,
                eta,
            }))
        }
        "extended_standard" => {
            let eps = get_f64("epsilon", 0.01)?;
            let delta = get_f64("delta", 0.001)?;
            Ok(Box::new(kdmd::ExtendedStandardMap {
                epsilon: eps,
                delta,
            }))
        }
        "henon" => {
            let a = get_f64("a", 1.4)?;
            let b = get_f64("b", 0.3)?;
            Ok(Box::new(kdmd::HenonMap { a, b }))
        }
        "logistic" => {
            let r = get_f64("r", 3.9)?;
            Ok(Box::new(kdmd::LogisticMap { r }))
        }
        _ => Err(PyValueError::new_err(format!("unknown map: '{map_name}'"))),
    }
}

// ============================================================================
// Module definition
// ============================================================================

/// Koopman DMD - Dynamic Mode Decomposition with Koopman operator theory.
///
/// Classes
/// -------
/// DMD : Standard DMD decomposition
/// HankelDMD : Hankel (time-delay embedding) DMD
/// GLA : Generalized Laplace Analysis
///
/// Functions
/// ---------
/// generate_trajectory : Generate map trajectories
/// harmonic_time_average : Compute HTA
/// mesochronic_compute : Mesochronic Harmonic Plot grid computation
/// classify_phase_space : Classify by HTA magnitude
/// hta_convergence : HTA convergence analysis
#[pymodule]
fn koopman_dmd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DMD>()?;
    m.add_class::<HankelDMD>()?;
    m.add_class::<GLA>()?;
    m.add_function(wrap_pyfunction!(generate_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_time_average, m)?)?;
    m.add_function(wrap_pyfunction!(mesochronic_compute, m)?)?;
    m.add_function(wrap_pyfunction!(classify_phase_space, m)?)?;
    m.add_function(wrap_pyfunction!(hta_convergence, m)?)?;
    Ok(())
}
