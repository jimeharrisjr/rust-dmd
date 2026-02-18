#' Harmonic Time Average
#'
#' @param map_name Character map name.
#' @param initial_condition Numeric vector.
#' @param observable Character observable name.
#' @param omega Numeric frequency.
#' @param n_iter Integer iterations.
#' @param ... Map parameters.
#' @return List with magnitude, phase, hta_re, hta_im.
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
#' @export
hta_convergence <- function(map_name, initial_condition, observable = "sin_pi",
                             omega = 0.1, n_iter = 10000, ...) {
  params <- list(...)
  rust_hta_convergence(map_name, initial_condition, observable,
                        omega, as.integer(n_iter), params)
}
