test_that("DMD basic decomposition works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  expect_s3_class(d, "dmd")
  expect_true(d$rank > 0)
  expect_equal(d$data_dim, c(2L, 100L))
  expect_false(d$center)
  expect_equal(d$dt, 1.0)
})

test_that("DMD with rank specification works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X, rank = 1)
  expect_equal(d$rank, 1L)
})

test_that("DMD with centering works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals) + 5, cos(t_vals) + 3)

  d <- dmd(X, center = TRUE)
  expect_true(d$center)
})

test_that("DMD eigenvalues are returned correctly", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  expect_equal(length(d$eigenvalues_re), d$rank)
  expect_equal(length(d$eigenvalues_im), d$rank)
})

test_that("DMD predict works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  pred <- predict(d, n_ahead = 10)
  expect_true(is.matrix(pred))
  expect_equal(nrow(pred), 2)
  expect_equal(ncol(pred), 10)
})

test_that("DMD predict with initial condition works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  pred <- predict(d, n_ahead = 5, x0 = c(0, 1))
  expect_equal(dim(pred), c(2, 5))
})

test_that("DMD reconstruct works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  recon <- dmd_reconstruct(d, n_steps = 100)
  expect_equal(dim(recon), c(2, 100))

  rmse <- sqrt(mean((recon - X)^2))
  expect_lt(rmse, 0.5)
})

test_that("DMD spectrum returns data frame", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  spec <- dmd_spectrum(d)
  expect_true(is.data.frame(spec))
  expect_equal(nrow(spec), d$rank)
  expect_true("magnitude" %in% names(spec))
  expect_true("frequency" %in% names(spec))
  expect_true("stability" %in% names(spec))
})

test_that("DMD stability analysis works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  stab <- dmd_stability(d)
  expect_true(is.logical(stab$is_stable))
  expect_true(is.numeric(stab$spectral_radius))
  expect_gt(stab$spectral_radius, 0)
})

test_that("DMD error metrics work", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  err <- dmd_error(d)
  expect_true(err$rmse >= 0)
  expect_true(err$mae >= 0)
  expect_true(err$relative_error >= 0)
})

test_that("DMD dominant modes works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  dom <- dmd_dominant_modes(d, n = 1)
  expect_equal(length(dom), 1)
  expect_true(dom[1] >= 1 && dom[1] <= d$rank)
})

test_that("DMD residual works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X)
  res <- dmd_residual(d)
  expect_true(res$residual_norm >= 0)
  expect_true(res$residual_relative >= 0)
})

test_that("DMD print works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))
  d <- dmd(X)
  expect_output(print(d), "DMD")
})

test_that("DMD with lifting works", {
  t_vals <- seq(0, 2 * pi, length.out = 100)
  X <- rbind(sin(t_vals), cos(t_vals))

  d <- dmd(X, lifting = "polynomial", lifting_param = 2)
  expect_true(d$rank > 0)
})
