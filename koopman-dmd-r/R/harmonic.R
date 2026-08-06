#' Harmonic Time Average
#'
#' @param map_name Character map name.
#' @param initial_condition Numeric vector.
#' @param observable Character observable name.
#' @param omega Numeric frequency.
#' @param n_iter Integer iterations.
#' @param ... Map parameters.
#' @return List with magnitude, phase, hta_re, hta_im.
#'
#' @examples
#' # Harmonic time average of an orbit of the Chirikov standard map
#' harmonic_time_average("standard", c(0.1, 0.2), "sin_pi", 0.1, 500)
#' @export
harmonic_time_average <- function(map_name, initial_condition, observable = "sin_pi",
                                   omega = 0.1, n_iter = 10000, ...) {
  params <- list(...)
  rust_harmonic_time_average(map_name, initial_condition, observable,
                              omega, as.integer(n_iter), params)
}

#' Mesochronic harmonic plot computation
#'
#' @param map_name Character map name.
#' @param x_range Numeric vector c(min, max).
#' @param y_range Numeric vector c(min, max).
#' @param resolution Integer grid resolution.
#' @param observable Character observable name.
#' @param omega Numeric frequency.
#' @param n_iter Integer iterations.
#' @param ... Map parameters.
#' @return List with hta_matrix, phase_matrix, x_coords, y_coords.
#'
#' @examples
#' # Mesochronic plot over a coarse grid. Raise `resolution` and `n_iter`
#' # for publication-quality figures; both cost time roughly linearly.
#' mhp <- mesochronic_compute("standard", c(0, 1), c(0, 1), 10, "sin_pi", 0.1, 100)
#' str(mhp)
#' @export
mesochronic_compute <- function(map_name, x_range = c(0, 1), y_range = c(0, 1),
                                 resolution = 100, observable = "sin_pi",
                                 omega = 0.1, n_iter = 30000, ...) {
  params <- list(...)
  rust_mesochronic_compute(map_name, x_range, y_range, as.integer(resolution),
                            observable, omega, as.integer(n_iter), params)
}

#' Classify phase space points by HTA magnitude
#'
#' @param hta_magnitudes Numeric vector of |HTA| values.
#' @param resonating_threshold Threshold for resonating. Default 0.01.
#' @param chaotic_threshold Threshold for chaotic. Default 0.0001.
#' @return Integer vector (1=resonating, 2=chaotic, 3=non-resonating).
#'
#' @examples
#' # Classify orbits from their HTA magnitudes
#' mags <- c(0.5, 0.001, 1e-06, 0.1)
#' classify_phase_space(mags)
#'
#' # Thresholds are adjustable
#' classify_phase_space(mags, resonating_threshold = 0.1, chaotic_threshold = 0.001)
#' @export
classify_phase_space <- function(hta_magnitudes,
                                  resonating_threshold = 0.01,
                                  chaotic_threshold = 0.0001) {
  rust_classify_phase_space(hta_magnitudes, resonating_threshold, chaotic_threshold)
}

#' HTA convergence analysis
#'
#' @param map_name Character map name.
#' @param initial_condition Numeric vector.
#' @param observable Character observable name.
#' @param omega Numeric frequency.
#' @param n_iter Integer iterations.
#' @param ... Map parameters.
#' @return List with times, hta_magnitudes, convergence_rate, dynamics_type.
#'
#' @examples
#' # How the time average converges along the orbit
#' conv <- hta_convergence("standard", c(0.1, 0.2), "sin_pi", 0.1, 500)
#' str(conv)
#' @export
hta_convergence <- function(map_name, initial_condition, observable = "sin_pi",
                             omega = 0.1, n_iter = 10000, ...) {
  params <- list(...)
  rust_hta_convergence(map_name, initial_condition, observable,
                        omega, as.integer(n_iter), params)
}
