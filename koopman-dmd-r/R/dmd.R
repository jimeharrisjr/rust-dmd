#' Dynamic Mode Decomposition
#'
#' Perform Dynamic Mode Decomposition on time-series data.
#'
#' @param X Numeric matrix (n_vars x n_time).
#' @param rank Integer truncation rank, or NULL for automatic.
#' @param center Logical, center data by subtracting row means.
#' @param dt Numeric time step.
#' @param lifting Character lifting type or NULL.
#' @param lifting_param Integer lifting parameter or NULL.
#' @return An S3 object of class "dmd".
#' @export
dmd <- function(X, rank = NULL, center = FALSE, dt = 1.0,
                lifting = NULL, lifting_param = NULL) {
  if (!is.matrix(X)) X <- as.matrix(X)
  res <- rust_dmd(X, rank, center, dt, lifting, lifting_param)
  structure(res, class = "dmd")
}

#' @export
print.dmd <- function(x, ...) {
  cat(sprintf("DMD(rank=%d, data_dim=(%d, %d), center=%s)\n",
              x$rank, x$data_dim[1], x$data_dim[2],
              ifelse(x$center, "TRUE", "FALSE")))
  invisible(x)
}

#' @export
summary.dmd <- function(object, ...) {
  cat("Dynamic Mode Decomposition\n")
  cat(sprintf("  Rank: %d\n", object$rank))
  cat(sprintf("  Data dimensions: %d variables x %d time steps\n",
              object$data_dim[1], object$data_dim[2]))
  cat(sprintf("  Centered: %s\n", object$center))
  cat(sprintf("  Time step: %g\n", object$dt))
  cat(sprintf("  Singular values: %s\n",
              paste(round(object$singular_values, 4), collapse = ", ")))

  # Eigenvalue magnitudes
  mags <- sqrt(object$eigenvalues_re^2 + object$eigenvalues_im^2)
  cat(sprintf("  Eigenvalue magnitudes: %s\n",
              paste(round(mags, 4), collapse = ", ")))
  invisible(object)
}

#' Predict from DMD model
#'
#' @param object A dmd object.
#' @param n_ahead Number of steps to predict.
#' @param x0 Optional initial condition vector.
#' @param method "modes" (default) or "matrix".
#' @param ... Additional arguments (ignored).
#' @return Numeric matrix of predictions.
#' @export
predict.dmd <- function(object, n_ahead = 10, x0 = NULL,
                        method = c("modes", "matrix"), ...) {
  method <- match.arg(method)
  rust_dmd_predict(object$`_result_ptr`, as.integer(n_ahead), x0, method)
}

#' Reconstruct data from DMD modes
#'
#' @param object A dmd object.
#' @param n_steps Number of time steps. Defaults to original length.
#' @param modes Integer vector of mode indices (1-based), or NULL for all.
#' @return Numeric matrix.
#' @export
dmd_reconstruct <- function(object, n_steps = NULL, modes = NULL) {
  if (is.null(n_steps)) n_steps <- object$data_dim[2]
  rust_dmd_reconstruct(object$`_result_ptr`, as.integer(n_steps), modes)
}

#' DMD spectrum analysis
#'
#' @param object A dmd object.
#' @param dt Time step. Uses stored dt by default.
#' @return Data frame with mode information.
#' @export
dmd_spectrum <- function(object, dt = NULL) {
  if (is.null(dt)) dt <- object$dt
  res <- rust_dmd_spectrum(object$`_result_ptr`, dt)
  as.data.frame(res)
}

#' DMD stability analysis
#'
#' @param object A dmd object.
#' @param tol Tolerance for marginal classification.
#' @return List with stability information.
#' @export
dmd_stability <- function(object, tol = 1e-6) {
  rust_dmd_stability(object$`_result_ptr`, tol)
}

#' DMD reconstruction error
#'
#' @param object A dmd object.
#' @return List with error metrics (rmse, mae, mape, relative_error).
#' @export
dmd_error <- function(object) {
  rust_dmd_error(object$`_result_ptr`, object$`_x_ptr`)
}

#' DMD dominant modes
#'
#' @param object A dmd object.
#' @param n Number of modes.
#' @param criterion "amplitude", "energy", or "stability".
#' @return Integer vector of 1-based mode indices.
#' @export
dmd_dominant_modes <- function(object, n = 3,
                               criterion = c("amplitude", "energy", "stability")) {
  criterion <- match.arg(criterion)
  rust_dmd_dominant_modes(object$`_result_ptr`, as.integer(n), criterion)
}

#' DMD residual analysis
#'
#' @param object A dmd object.
#' @return List with residual_norm and residual_relative.
#' @export
dmd_residual <- function(object) {
  rust_dmd_residual(object$`_result_ptr`, object$`_x_ptr`)
}
