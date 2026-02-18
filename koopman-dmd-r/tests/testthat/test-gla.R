test_that("GLA basic works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))

  g <- gla(y, n_eigenvalues = 2)
  expect_s3_class(g, "gla")
  expect_equal(g$n_obs, 2L)
  expect_equal(g$n_time, 200L)
})

test_that("GLA eigenvalues returned", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))

  g <- gla(y, n_eigenvalues = 2)
  expect_equal(length(g$eigenvalues_re), 2)
  expect_equal(length(g$eigenvalues_im), 2)
})

test_that("GLA convergence info", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))

  g <- gla(y, n_eigenvalues = 2)
  expect_true(is.logical(g$convergence))
  expect_true(is.numeric(g$residuals))
})

test_that("GLA predict works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))

  g <- gla(y, n_eigenvalues = 2)
  pred <- predict(g, n_ahead = 10)
  expect_true(is.matrix(pred))
  expect_equal(nrow(pred), 2)
})

test_that("GLA reconstruct works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))

  g <- gla(y, n_eigenvalues = 2)
  recon <- gla_reconstruct(g)
  expect_true(is.matrix(recon))
  expect_equal(nrow(recon), 2)
})

test_that("GLA with known eigenvalues", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))

  g <- gla(y, eigenvalues = complex(real = c(0.99, 0.99), imaginary = c(0.1, -0.1)))
  expect_equal(length(g$eigenvalues_re), 2)
})

test_that("GLA print works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- rbind(sin(t_vals), cos(t_vals))
  g <- gla(y, n_eigenvalues = 2)
  expect_output(print(g), "GLA")
})
