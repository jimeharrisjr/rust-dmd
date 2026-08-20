# Simulate x_{t+1} = A0 x_t + B0 u_t with a persistently exciting input,
# returning the pair matrices (X1, X2, U).
make_forced <- function(m = 120) {
  A0 <- matrix(c(0.9, 0, 0.1, 0.8), 2, 2)
  B0 <- matrix(c(0.5, 1), 2, 1)
  X1 <- matrix(0, 2, m)
  X2 <- matrix(0, 2, m)
  U <- matrix(0, 1, m)
  x <- c(1, -0.5)
  for (t in seq_len(m)) {
    u_t <- sin(0.7 * (t - 1)) + 0.5 * cos(2.3 * (t - 1) + 1)
    X1[, t] <- x
    U[, t] <- u_t
    x <- as.numeric(A0 %*% x + B0 * u_t)
    X2[, t] <- x
  }
  list(X1 = X1, X2 = X2, U = U, A0 = A0, B0 = B0)
}

test_that("DMDc recovers A and B jointly", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  expect_s3_class(fit, "dmdc")
  expect_equal(fit$a, sim$A0, tolerance = 1e-9)
  expect_equal(fit$b, sim$B0, tolerance = 1e-9)
  expect_equal(fit$rank_input, 3L)
  expect_equal(fit$rank_output, 2L)
  expect_equal(fit$n_states, 2L)
  expect_equal(fit$n_inputs, 1L)
  # Full-order default: basis is the identity and a_tilde = a
  expect_equal(fit$basis, diag(2))
  expect_equal(fit$a_tilde, fit$a)
})

test_that("DMDc eigenvalues match the true dynamics", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  mags <- sort(sqrt(fit$eigenvalues_re^2 + fit$eigenvalues_im^2))
  expect_equal(mags, c(0.8, 0.9), tolerance = 1e-9)
})

test_that("DMDc with known B pins the input matrix", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 2, known_B = sim$B0)
  expect_equal(fit$a, sim$A0, tolerance = 1e-9)
  expect_identical(fit$b, sim$B0)
})

test_that("DMDc fits autonomous pairs with U = NULL", {
  A0 <- matrix(c(0.95, 0, 0.02, 0.85), 2, 2)
  cols <- 0
  X1 <- matrix(0, 2, 80)
  X2 <- matrix(0, 2, 80)
  for (start in list(c(1, 0.5), c(-0.3, 1.2))) {
    x <- start
    for (i in seq_len(40)) {
      cols <- cols + 1
      X1[, cols] <- x
      x <- as.numeric(A0 %*% x)
      X2[, cols] <- x
    }
  }
  fit <- dmdc(X1, X2, rank_input = 2)
  expect_equal(fit$a, A0, tolerance = 1e-9)
  expect_equal(fit$n_inputs, 0L)
})

test_that("DMDc output projection produces reduced operators", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3, rank_output = 2)
  expect_equal(dim(fit$a_tilde), c(2L, 2L))
  expect_equal(dim(fit$b_tilde), c(2L, 1L))
  expect_equal(dim(fit$basis), c(2L, 2L))
  # Basis columns are orthonormal
  expect_equal(t(fit$basis) %*% fit$basis, diag(2), tolerance = 1e-12)
})

test_that("DMDc accepts a vector U as a single input row", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, as.numeric(sim$U), rank_input = 3)
  expect_equal(fit$b, sim$B0, tolerance = 1e-9)
})

test_that("predict.dmdc replays the training inputs exactly", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  pred <- predict(fit, U = sim$U)
  expect_equal(pred, sim$X2, tolerance = 1e-7)
})

test_that("predict.dmdc zero-input response follows A alone", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  x0 <- c(1, 1)
  pred <- predict(fit, x0 = x0, n_ahead = 5)
  expect_equal(dim(pred), c(2L, 5L))
  x <- x0
  for (t in seq_len(5)) {
    x <- as.numeric(sim$A0 %*% x)
    expect_equal(pred[, t], x, tolerance = 1e-7)
  }
})

test_that("predict.dmdc validates its inputs", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  expect_error(predict(fit), "either U or n_ahead")
  expect_error(predict(fit, U = matrix(0, 2, 5)), "rows")
  expect_error(predict(fit, U = sim$U, n_ahead = 3), "does not match")
  expect_error(predict(fit, U = sim$U, x0 = c(1, 2, 3)), "length")
})

test_that("DMDc rejects invalid inputs", {
  sim <- make_forced()
  expect_error(dmdc(sim$X1, sim$X2[, -1], sim$U))
  expect_error(dmdc(sim$X1, sim$X2, sim$U[, -1, drop = FALSE]))
  expect_error(dmdc(sim$X1, sim$X2, sim$U, known_B = matrix(0, 3, 1)))
  expect_error(dmdc(sim$X1, sim$X2, sim$U, dt = 0))
})

test_that("dmdc_stability classifies the identified operator", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  stab <- dmdc_stability(fit)
  expect_true(stab$is_stable)
  expect_false(stab$is_unstable)
  expect_equal(stab$spectral_radius, 0.9, tolerance = 1e-9)
})

test_that("dmdc_spectrum returns per-mode information", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  spec <- dmdc_spectrum(fit)
  expect_s3_class(spec, "data.frame")
  expect_equal(nrow(spec), 2L)
  expect_equal(sort(spec$magnitude), c(0.8, 0.9), tolerance = 1e-9)
  expect_true(all(spec$amplitude == 0))
})

test_that("print and summary methods work for dmdc", {
  sim <- make_forced()
  fit <- dmdc(sim$X1, sim$X2, sim$U, rank_input = 3)
  expect_output(print(fit), "DMDc")
  expect_output(summary(fit), "Dynamic Mode Decomposition with Control")
})
