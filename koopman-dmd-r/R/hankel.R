#' Hankel-DMD (Time-Delay Embedding DMD)
#'
#' @param y Numeric matrix (n_obs x n_time).
#' @param delays Integer number of delays, or NULL for automatic.
#' @param rank Integer truncation rank, or NULL for automatic.
#' @param dt Numeric time step.
#' @return An S3 object of class "hankel_dmd".
#' @export
hankel_dmd <- function(y, delays = NULL, rank = NULL, dt = 1.0) {
  if (!is.matrix(y)) y <- as.matrix(y)
  if (nrow(y) > ncol(y)) y <- t(y)  # Ensure (n_obs x n_time)
  res <- rust_hankel_dmd(y, delays, rank, dt)
  structure(res, class = c("hankel_dmd", "dmd"))
}

#' @export
print.hankel_dmd <- function(x, ...) {
  cat(sprintf("HankelDMD(rank=%d, delays=%d, n_obs=%d)\n",
              x$rank, x$delays, x$n_obs))
  invisible(x)
}

#' Reconstruct from Hankel-DMD
#'
#' @param object A hankel_dmd object.
#' @param n_steps Number of time steps.
#' @return Numeric matrix.
#' @export
hankel_reconstruct <- function(object, n_steps) {
  rust_hankel_reconstruct(object$`_result_ptr`, as.integer(n_steps))
}

#' Predict from Hankel-DMD
#'
#' @param object A hankel_dmd object.
#' @param n_ahead Number of steps to predict.
#' @param ... Additional arguments (ignored).
#' @return Numeric matrix.
#' @export
predict.hankel_dmd <- function(object, n_ahead = 10, ...) {
  rust_hankel_predict(object$`_result_ptr`, as.integer(n_ahead))
}
