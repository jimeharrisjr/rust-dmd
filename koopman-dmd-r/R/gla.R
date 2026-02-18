#' Generalized Laplace Analysis
#'
#' @param y Numeric matrix (n_obs x n_time).
#' @param eigenvalues Complex vector of known eigenvalues, or NULL.
#' @param n_eigenvalues Number of eigenvalues to estimate.
#' @param tol Convergence tolerance.
#' @param max_iter Maximum iterations, or NULL.
#' @return An S3 object of class "gla".
#' @export
gla <- function(y, eigenvalues = NULL, n_eigenvalues = 5,
                tol = 1e-6, max_iter = NULL) {
  if (!is.matrix(y)) y <- as.matrix(y)
  if (nrow(y) > ncol(y)) y <- t(y)

  eig_re <- NULL
  eig_im <- NULL
  if (!is.null(eigenvalues)) {
    eig_re <- Re(eigenvalues)
    eig_im <- Im(eigenvalues)
  }

  res <- rust_gla(y, eig_re, eig_im, as.integer(n_eigenvalues),
                  tol, max_iter)
  structure(res, class = "gla")
}

#' @export
print.gla <- function(x, ...) {
  cat(sprintf("GLA(n_eigenvalues=%d, n_obs=%d, n_time=%d)\n",
              length(x$eigenvalues_re), x$n_obs, x$n_time))
  invisible(x)
}

#' Reconstruct from GLA
#'
#' @param object A gla object.
#' @param modes_to_use Integer vector of mode indices (1-based), or NULL.
#' @return Numeric matrix.
#' @export
gla_reconstruct <- function(object, modes_to_use = NULL) {
  rust_gla_reconstruct(object$`_result_ptr`, modes_to_use)
}

#' Predict from GLA
#'
#' @param object A gla object.
#' @param n_ahead Number of steps to predict.
#' @param ... Additional arguments (ignored).
#' @return Numeric matrix.
#' @export
predict.gla <- function(object, n_ahead = 10, ...) {
  rust_gla_predict(object$`_result_ptr`, as.integer(n_ahead))
}
