#' Generate a trajectory from a built-in map
#'
#' @param map_name Character: "standard", "froeschle", "extended_standard",
#'   "henon", or "logistic".
#' @param initial_condition Numeric vector.
#' @param n_iter Integer number of iterations.
#' @param ... Map parameters passed as named arguments.
#' @return Numeric matrix (n_dim x n_iter+1).
#' @export
generate_trajectory <- function(map_name, initial_condition, n_iter, ...) {
  params <- list(...)
  rust_generate_trajectory(map_name, initial_condition, as.integer(n_iter), params)
}

#' Standard map (Chirikov)
#' @param state Numeric vector c(x, y).
#' @param epsilon Perturbation parameter.
#' @return Updated state vector.
#' @export
standard_map <- function(state, epsilon = 0.12) {
  traj <- rust_generate_trajectory("standard", state, 1L, list(epsilon = epsilon))
  traj[, 2]
}

#' Froeschle map (4D coupled standard maps)
#' @param state Numeric vector of length 4.
#' @param epsilon1 First perturbation.
#' @param epsilon2 Second perturbation.
#' @param eta Coupling parameter.
#' @return Updated state vector.
#' @export
froeschle_map <- function(state, epsilon1 = 0.02, epsilon2 = 0.02, eta = 0.01) {
  traj <- rust_generate_trajectory("froeschle", state, 1L,
                                    list(epsilon1 = epsilon1, epsilon2 = epsilon2, eta = eta))
  traj[, 2]
}

#' Extended standard map (3D)
#' @param state Numeric vector of length 3.
#' @param epsilon Perturbation parameter.
#' @param delta Coupling parameter.
#' @return Updated state vector.
#' @export
extended_standard_map <- function(state, epsilon = 0.01, delta = 0.001) {
  traj <- rust_generate_trajectory("extended_standard", state, 1L,
                                    list(epsilon = epsilon, delta = delta))
  traj[, 2]
}

#' Henon map (2D dissipative)
#' @param state Numeric vector c(x, y).
#' @param a Parameter a.
#' @param b Parameter b.
#' @return Updated state vector.
#' @export
henon_map <- function(state, a = 1.4, b = 0.3) {
  traj <- rust_generate_trajectory("henon", state, 1L, list(a = a, b = b))
  traj[, 2]
}

#' Logistic map (1D)
#' @param state Numeric scalar.
#' @param r Growth rate parameter.
#' @return Updated state value.
#' @export
logistic_map <- function(state, r = 3.9) {
  traj <- rust_generate_trajectory("logistic", state, 1L, list(r = r))
  traj[, 2]
}
