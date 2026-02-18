test_that("Hankel-DMD basic works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- matrix(sin(t_vals), nrow = 1)

  h <- hankel_dmd(y)
  expect_s3_class(h, "hankel_dmd")
  expect_true(h$rank > 0)
  expect_true(h$delays > 0)
  expect_equal(h$n_obs, 1L)
})

test_that("Hankel-DMD with specified delays works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- matrix(sin(t_vals), nrow = 1)

  h <- hankel_dmd(y, delays = 10)
  expect_equal(h$delays, 10L)
})

test_that("Hankel-DMD eigenvalues are returned", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- matrix(sin(t_vals), nrow = 1)

  h <- hankel_dmd(y, delays = 10)
  expect_equal(length(h$eigenvalues_re), h$rank)
  expect_equal(length(h$eigenvalues_im), h$rank)
})

test_that("Hankel-DMD reconstruct works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- matrix(sin(t_vals), nrow = 1)

  h <- hankel_dmd(y, delays = 10)
  recon <- hankel_reconstruct(h, 50)
  expect_true(is.matrix(recon))
  expect_equal(nrow(recon), 1)
})

test_that("Hankel-DMD predict works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- matrix(sin(t_vals), nrow = 1)

  h <- hankel_dmd(y, delays = 10)
  pred <- predict(h, n_ahead = 20)
  expect_true(is.matrix(pred))
  expect_equal(nrow(pred), 1)
})

test_that("Hankel-DMD print works", {
  t_vals <- seq(0, 4 * pi, length.out = 200)
  y <- matrix(sin(t_vals), nrow = 1)
  h <- hankel_dmd(y, delays = 10)
  expect_output(print(h), "HankelDMD")
})
