test_that("HTA basic works", {
  result <- harmonic_time_average("standard", c(0.1, 0.2), "sin_pi", 0.1, 1000)
  expect_true(is.list(result))
  expect_true(result$magnitude >= 0)
  expect_true(is.numeric(result$phase))
})

test_that("HTA different observables work", {
  for (obs in c("identity", "sin_pi", "cos_pi", "sin_2pi", "cos_2pi")) {
    result <- harmonic_time_average("standard", c(0.1, 0.2), obs, 0.1, 500)
    expect_true(result$magnitude >= 0)
  }
})

test_that("HTA invalid observable errors", {
  expect_error(harmonic_time_average("standard", c(0.1, 0.2), "invalid", 0.1, 100))
})

test_that("Mesochronic compute works", {
  result <- mesochronic_compute("standard", c(0, 1), c(0, 1), 10, "sin_pi", 0.1, 100)
  expect_true(is.matrix(result$hta_matrix))
  expect_equal(dim(result$hta_matrix), c(10, 10))
  expect_equal(dim(result$phase_matrix), c(10, 10))
  expect_equal(length(result$x_coords), 10)
  expect_equal(length(result$y_coords), 10)
})

test_that("Mesochronic compute with params works", {
  result <- mesochronic_compute("standard", c(0, 1), c(0, 1), 5, "cos_pi", 0.2, 50,
                                 epsilon = 0.3)
  expect_equal(dim(result$hta_matrix), c(5, 5))
})

test_that("Classify phase space works", {
  mags <- c(0.5, 0.001, 1e-6, 0.1)
  labels <- classify_phase_space(mags)
  expect_equal(length(labels), 4)
  expect_true(all(labels %in% c(1, 2, 3)))
})

test_that("Classify with custom thresholds", {
  mags <- c(0.5, 0.005, 1e-6)
  labels <- classify_phase_space(mags, resonating_threshold = 0.1, chaotic_threshold = 0.001)
  expect_equal(length(labels), 3)
})

test_that("HTA convergence works", {
  result <- hta_convergence("standard", c(0.1, 0.2), "sin_pi", 0.1, 1000)
  expect_true(is.list(result))
  expect_true("times" %in% names(result))
  expect_true("hta_magnitudes" %in% names(result))
  expect_true("dynamics_type" %in% names(result))
  expect_equal(length(result$times), length(result$hta_magnitudes))
})
