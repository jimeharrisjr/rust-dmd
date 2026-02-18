test_that("DMD on map trajectory works", {
  traj <- generate_trajectory("standard", c(0.1, 0.2), 200)
  d <- dmd(traj, rank = 5)
  recon <- dmd_reconstruct(d, n_steps = ncol(traj))
  expect_equal(dim(recon), dim(traj))
})

test_that("Hankel-DMD on logistic map works", {
  traj <- generate_trajectory("logistic", 0.5, 300, r = 3.5)
  h <- hankel_dmd(traj, delays = 10, rank = 5)
  expect_equal(h$rank, 5L)
})
