test_that("Standard map trajectory works", {
  traj <- generate_trajectory("standard", c(0.1, 0.2), 100)
  expect_true(is.matrix(traj))
  expect_equal(nrow(traj), 2)
  expect_equal(ncol(traj), 101)
})

test_that("Standard map with epsilon works", {
  traj <- generate_trajectory("standard", c(0.1, 0.2), 50, epsilon = 0.3)
  expect_equal(dim(traj), c(2, 51))
})

test_that("Henon map works", {
  traj <- generate_trajectory("henon", c(0, 0), 100, a = 1.4, b = 0.3)
  expect_equal(dim(traj), c(2, 101))
})

test_that("Logistic map works", {
  traj <- generate_trajectory("logistic", 0.5, 100, r = 3.9)
  expect_equal(dim(traj), c(1, 101))
  expect_true(all(traj >= 0 & traj <= 1))
})

test_that("Froeschle map works", {
  traj <- generate_trajectory("froeschle", c(0.1, 0.2, 0.3, 0.4), 50)
  expect_equal(dim(traj), c(4, 51))
})

test_that("Extended standard map works", {
  traj <- generate_trajectory("extended_standard", c(0.1, 0.2, 0.3), 50)
  expect_equal(dim(traj), c(3, 51))
})

test_that("Unknown map errors", {
  expect_error(generate_trajectory("unknown", c(0.1), 10))
})

test_that("Single-step map functions work", {
  s <- standard_map(c(0.1, 0.2))
  expect_equal(length(s), 2)

  h <- henon_map(c(0, 0))
  expect_equal(length(h), 2)

  l <- logistic_map(0.5)
  expect_equal(length(l), 1)
})
