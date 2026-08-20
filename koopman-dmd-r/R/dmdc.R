#' Dynamic Mode Decomposition with Control (DMDc)
#'
#' Identify the forced linear system \eqn{x_{t+1} = A x_t + B u_t} from
#' snapshot pairs and control inputs, following Proctor, Brunton and Kutz
#' (2016). Unlike \code{\link{dmd}}, which takes one contiguous trajectory,
#' \code{dmdc} takes explicit pair matrices: \code{X1} holds states at time
#' \eqn{t}, \code{X2} the states one step later, and \code{U} the control
#' input applied during each transition. Columns may therefore come from many
#' concatenated trajectories.
#'
#' Two identification modes are available. With \code{known_B = NULL} (the
#' default), \code{A} and \code{B} are solved jointly from the stacked
#' regression \eqn{[A~B] = X_2 [X_1; U]^+}; this requires the input to be
#' persistently exciting and exogenous (not state feedback). When the input
#' coupling is known by construction, pass it as \code{known_B} and only
#' \code{A} is estimated.
#'
#' @param X1 Numeric matrix of states at time t (n_states x n_pairs).
#' @param X2 Numeric matrix of states at time t+1 (n_states x n_pairs).
#' @param U Numeric matrix of control inputs during each transition
#'   (n_inputs x n_pairs), or NULL for an autonomous multi-trajectory fit.
#'   A vector is taken as a single input row.
#' @param rank_input Integer truncation rank for the regression-input SVD,
#'   or NULL for automatic (99 percent cumulative variance).
#' @param rank_output Integer rank of the output basis (SVD of \code{X2}),
#'   or NULL to keep everything full-order.
#' @param dt Numeric time step between snapshot pairs.
#' @param known_B Known input matrix B (n_states x n_inputs), or NULL to
#'   estimate B jointly with A.
#' @return An S3 object of class "dmdc" with components \code{a}, \code{b},
#'   \code{a_tilde}, \code{b_tilde}, \code{basis}, \code{eigenvalues_re},
#'   \code{eigenvalues_im}, \code{singular_values}, \code{rank_input},
#'   \code{rank_output}, \code{dt}, \code{n_states}, and \code{n_inputs}.
#'
#' @references Proctor, J. L., Brunton, S. L., and Kutz, J. N. (2016).
#' Dynamic Mode Decomposition with Control. \emph{SIAM Journal on Applied
#' Dynamical Systems}, 15(1), 142-161. \doi{10.1137/15M1013857}
#'
#' @seealso \code{\link{dmd}} for autonomous systems,
#'   \code{\link{predict.dmdc}} to simulate the identified system.
#'
#' @examples
#' # Simulate x_{t+1} = A0 x_t + B0 u_t with a persistently exciting input
#' A0 <- matrix(c(0.9, 0, 0.1, 0.8), 2, 2)
#' B0 <- matrix(c(0.5, 1), 2, 1)
#' m <- 120
#' X1 <- matrix(0, 2, m)
#' X2 <- matrix(0, 2, m)
#' U <- matrix(0, 1, m)
#' x <- c(1, -0.5)
#' for (t in seq_len(m)) {
#'   u_t <- sin(0.7 * (t - 1)) + 0.5 * cos(2.3 * (t - 1) + 1)
#'   X1[, t] <- x
#'   U[, t] <- u_t
#'   x <- as.numeric(A0 %*% x + B0 * u_t)
#'   X2[, t] <- x
#' }
#'
#' # Identify A and B jointly; both are recovered to machine precision
#' fit <- dmdc(X1, X2, U, rank_input = 3)
#' round(fit$a, 6)
#' round(fit$b, 6)
#'
#' # Known B: pin the input matrix and estimate only A
#' fit2 <- dmdc(X1, X2, U, rank_input = 2, known_B = B0)
#' round(fit2$a, 6)
#' @export
dmdc <- function(X1, X2, U = NULL, rank_input = NULL, rank_output = NULL,
                 dt = 1.0, known_B = NULL) {
  if (!is.matrix(X1)) X1 <- as.matrix(X1)
  if (!is.matrix(X2)) X2 <- as.matrix(X2)
  if (!is.null(U) && !is.matrix(U)) U <- matrix(U, nrow = 1)
  if (!is.null(known_B) && !is.matrix(known_B)) known_B <- as.matrix(known_B)
  res <- rust_dmdc(X1, X2, U, rank_input, rank_output, dt, known_B)
  structure(res, class = "dmdc")
}

#' @export
print.dmdc <- function(x, ...) {
  cat(sprintf("DMDc(n_states=%d, n_inputs=%d, rank_input=%d, rank_output=%d)\n",
              x$n_states, x$n_inputs, x$rank_input, x$rank_output))
  invisible(x)
}

#' @export
summary.dmdc <- function(object, ...) {
  cat("Dynamic Mode Decomposition with Control\n")
  cat(sprintf("  States: %d, inputs: %d\n", object$n_states, object$n_inputs))
  cat(sprintf("  Regression-input rank: %d\n", object$rank_input))
  cat(sprintf("  Output basis rank: %d\n", object$rank_output))
  cat(sprintf("  Time step: %g\n", object$dt))
  cat(sprintf("  Singular values: %s\n",
              paste(round(object$singular_values, 4), collapse = ", ")))
  mags <- sqrt(object$eigenvalues_re^2 + object$eigenvalues_im^2)
  cat(sprintf("  Eigenvalue magnitudes: %s\n",
              paste(round(mags, 4), collapse = ", ")))
  invisible(object)
}

#' DMDc stability analysis
#'
#' Classify the stability of the operator identified by \code{\link{dmdc}}
#' from the eigenvalues of \eqn{\tilde{A}}.
#'
#' @param object A dmdc object.
#' @param tol Tolerance for marginal classification.
#' @return List with stability information (\code{is_stable},
#'   \code{is_unstable}, \code{is_marginal}, \code{spectral_radius}).
#'
#' @examples
#' A0 <- matrix(c(0.9, 0, 0.1, 0.8), 2, 2)
#' B0 <- matrix(c(0.5, 1), 2, 1)
#' m <- 60
#' X1 <- matrix(0, 2, m); X2 <- matrix(0, 2, m); U <- matrix(0, 1, m)
#' x <- c(1, -0.5)
#' for (t in seq_len(m)) {
#'   u_t <- sin(0.7 * (t - 1)) + 0.5 * cos(2.3 * (t - 1) + 1)
#'   X1[, t] <- x
#'   U[, t] <- u_t
#'   x <- as.numeric(A0 %*% x + B0 * u_t)
#'   X2[, t] <- x
#' }
#' fit <- dmdc(X1, X2, U, rank_input = 3)
#'
#' dmdc_stability(fit)
#' @export
dmdc_stability <- function(object, tol = 1e-6) {
  rust_stability_from_eigenvalues(object$eigenvalues_re, object$eigenvalues_im, tol)
}

#' DMDc spectrum analysis
#'
#' Per-mode frequency, growth rate and stability for the operator identified
#' by \code{\link{dmdc}}. DMDc has no mode amplitudes, so the
#' \code{amplitude} column is reported as 0.
#'
#' @param object A dmdc object.
#' @param dt Time step. Uses the stored dt by default.
#' @return Data frame with mode information.
#'
#' @examples
#' A0 <- matrix(c(0.9, 0, 0.1, 0.8), 2, 2)
#' B0 <- matrix(c(0.5, 1), 2, 1)
#' m <- 60
#' X1 <- matrix(0, 2, m); X2 <- matrix(0, 2, m); U <- matrix(0, 1, m)
#' x <- c(1, -0.5)
#' for (t in seq_len(m)) {
#'   u_t <- sin(0.7 * (t - 1)) + 0.5 * cos(2.3 * (t - 1) + 1)
#'   X1[, t] <- x
#'   U[, t] <- u_t
#'   x <- as.numeric(A0 %*% x + B0 * u_t)
#'   X2[, t] <- x
#' }
#' fit <- dmdc(X1, X2, U, rank_input = 3)
#'
#' dmdc_spectrum(fit)
#' @export
dmdc_spectrum <- function(object, dt = NULL) {
  if (is.null(dt)) dt <- object$dt
  res <- rust_spectrum_from_eigenvalues(object$eigenvalues_re,
                                        object$eigenvalues_im, dt)
  as.data.frame(res)
}

#' Predict from a DMDc model
#'
#' Simulate the identified system \eqn{x_{t+1} = A x_t + B u_t} forward from
#' an initial state under a given control input sequence.
#'
#' @param object A dmdc object.
#' @param U Numeric matrix of control inputs (n_inputs x n_steps); the number
#'   of columns sets the prediction horizon. A vector is taken as a single
#'   input row. NULL applies zero input for \code{n_ahead} steps.
#' @param x0 Initial state vector. Defaults to the first stored snapshot.
#' @param n_ahead Number of steps when \code{U} is NULL; if both are given
#'   it must match \code{ncol(U)}.
#' @param ... Additional arguments (ignored).
#' @return Numeric matrix of predicted states \eqn{x_1 \ldots x_k}
#'   (n_states x k).
#'
#' @examples
#' A0 <- matrix(c(0.9, 0, 0.1, 0.8), 2, 2)
#' B0 <- matrix(c(0.5, 1), 2, 1)
#' m <- 120
#' X1 <- matrix(0, 2, m)
#' X2 <- matrix(0, 2, m)
#' U <- matrix(0, 1, m)
#' x <- c(1, -0.5)
#' for (t in seq_len(m)) {
#'   u_t <- sin(0.7 * (t - 1)) + 0.5 * cos(2.3 * (t - 1) + 1)
#'   X1[, t] <- x
#'   U[, t] <- u_t
#'   x <- as.numeric(A0 %*% x + B0 * u_t)
#'   X2[, t] <- x
#' }
#' fit <- dmdc(X1, X2, U, rank_input = 3)
#'
#' # Replaying the training inputs reproduces the observed successors
#' pred <- predict(fit, U = U)
#' max(abs(pred - X2))
#'
#' # Zero-input (free) response from a chosen state
#' free <- predict(fit, x0 = c(1, 1), n_ahead = 10)
#' dim(free)
#' @export
predict.dmdc <- function(object, U = NULL, x0 = NULL, n_ahead = NULL, ...) {
  q <- object$n_inputs
  if (is.null(U)) {
    if (is.null(n_ahead)) {
      stop("either U or n_ahead must be given")
    }
    U <- matrix(0, nrow = q, ncol = n_ahead)
  } else {
    if (!is.matrix(U)) U <- matrix(U, nrow = 1)
    if (nrow(U) != q) {
      stop(sprintf("U has %d rows, expected %d to match the fitted input matrix",
                   nrow(U), q))
    }
    if (!is.null(n_ahead) && n_ahead != ncol(U)) {
      stop(sprintf("n_ahead (%d) does not match the %d columns of U",
                   n_ahead, ncol(U)))
    }
  }
  k <- ncol(U)
  if (k < 1) stop("prediction horizon must be positive")

  if (is.null(x0)) x0 <- object$x_first
  if (length(x0) != object$n_states) {
    stop(sprintf("x0 has length %d, expected %d", length(x0), object$n_states))
  }

  out <- matrix(0, nrow = object$n_states, ncol = k)
  x <- as.numeric(x0)
  for (t in seq_len(k)) {
    x <- as.numeric(object$a %*% x)
    if (q > 0) x <- x + as.numeric(object$b %*% U[, t, drop = FALSE])
    out[, t] <- x
  }
  out
}
