#!/usr/bin/env Rscript
# Generate PNG plots for the documentation wiki
library(koopman.dmd)

outdir <- "docs/img"

col_actual  <- "#2196F3"
col_predict <- "#FF5722"
col_recon   <- "#4CAF50"
col_var2    <- "#9C27B0"

# =============================================================================
# Plot 1: Core DMD -- Predicted vs Actual
# =============================================================================
png(file.path(outdir, "dmd_predict_vs_actual.png"), width = 900, height = 600, res = 120)
dt <- 0.05; n_train <- 200; n_total <- 300
t_all <- seq(0, (n_total - 1) * dt, by = dt)
x1 <- exp(-0.05 * t_all) * sin(2 * pi * 0.5 * t_all)
x2 <- exp(-0.05 * t_all) * cos(2 * pi * 0.5 * t_all)
X_train <- rbind(x1[1:n_train], x2[1:n_train])

result <- dmd(X_train, rank = 2, dt = dt)
pred <- predict(result, n_ahead = n_total - 1)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
plot(t_all, x1, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "x1", main = "Variable 1")
lines(t_all[2:n_total], pred[1, ], col = col_predict, lwd = 2, lty = 2)
abline(v = t_all[n_train], lty = 3, col = "gray50")
legend("topright", c("Actual", "DMD Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)

plot(t_all, x2, type = "l", col = col_var2, lwd = 2,
     xlab = "Time", ylab = "x2", main = "Variable 2")
lines(t_all[2:n_total], pred[2, ], col = col_predict, lwd = 2, lty = 2)
abline(v = t_all[n_train], lty = 3, col = "gray50")
legend("topright", c("Actual", "DMD Predicted"),
       col = c(col_var2, col_predict), lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)
mtext("Core DMD: Damped Oscillator -- Predicted vs Actual", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 2: Eigenvalue Spectrum
# =============================================================================
png(file.path(outdir, "dmd_eigenvalues.png"), width = 900, height = 400, res = 120)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

theta <- seq(0, 2 * pi, length.out = 200)
plot(cos(theta), sin(theta), type = "l", col = "gray70", lwd = 1,
     xlab = "Re", ylab = "Im", asp = 1, main = "Eigenvalues in Complex Plane",
     xlim = c(-1.3, 1.3), ylim = c(-1.3, 1.3))
abline(h = 0, v = 0, col = "gray85")
points(result$eigenvalues_re, result$eigenvalues_im, pch = 19, col = col_predict, cex = 1.5)

spec <- dmd_spectrum(result)
barplot(spec$magnitude, names.arg = round(spec$frequency, 3),
        col = col_actual, border = NA,
        xlab = "Frequency (Hz)", ylab = "|lambda|", main = "Mode Magnitudes")
abline(h = 1.0, lty = 2, col = "gray50")
dev.off()

# =============================================================================
# Plot 3: Reconstruction Residuals
# =============================================================================
png(file.path(outdir, "dmd_residuals.png"), width = 900, height = 400, res = 120)
recon <- dmd_reconstruct(result)
err <- dmd_error(result)
t_train <- t_all[1:n_train]
res_x1 <- X_train[1, ] - recon[1, ]
res_x2 <- X_train[2, ] - recon[2, ]
par(mar = c(4, 4, 3, 1))
plot(t_train, res_x1, type = "l", col = col_actual, lwd = 1.5,
     xlab = "Time", ylab = "Residual",
     main = sprintf("DMD Reconstruction Residuals (RMSE = %.2e)", err$rmse),
     ylim = range(c(res_x1, res_x2)))
lines(t_train, res_x2, col = col_var2, lwd = 1.5)
abline(h = 0, lty = 2, col = "gray50")
legend("topright", c("x1", "x2"), col = c(col_actual, col_var2), lwd = 1.5, bg = "white")
dev.off()

# =============================================================================
# Plot 4: Extended DMD -- Lifting Comparison
# =============================================================================
png(file.path(outdir, "extended_dmd_lifting.png"), width = 900, height = 600, res = 120)
n3 <- 200; t3 <- seq(0, (n3 - 1) * 0.05, by = 0.05)
nonlinear <- sin(t3)^2
X3 <- matrix(nonlinear, nrow = 1)

res_std <- dmd(X3, dt = 0.05)
pred_std <- predict(res_std, n_ahead = n3 - 1)
err_std <- dmd_error(res_std)

res_lift <- dmd(X3, lifting = "polynomial", lifting_param = 2, dt = 0.05)
pred_lift <- predict(res_lift, n_ahead = n3 - 1)
err_lift <- dmd_error(res_lift)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
plot(t3, nonlinear, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "sin^2(t)",
     main = sprintf("Standard DMD (RMSE = %.4f)", err_std$rmse))
lines(t3[2:n3], pred_std[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("Actual", "Standard DMD"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2), bg = "white")

plot(t3, nonlinear, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "sin^2(t)",
     main = sprintf("Extended DMD with Polynomial Lifting (RMSE = %.4f)", err_lift$rmse))
lines(t3[2:n3], pred_lift[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Polynomial Lifted"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2), bg = "white")
mtext("Extended DMD: Lifting for Nonlinear Signals", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 5: Hankel-DMD -- Scalar Prediction
# =============================================================================
png(file.path(outdir, "hankel_dmd_prediction.png"), width = 900, height = 600, res = 120)
n4 <- 500; dt4 <- 0.02; t4 <- seq(0, (n4 - 1) * dt4, by = dt4)
scalar <- sin(2 * pi * 1.0 * t4) + 0.4 * sin(2 * pi * 3.0 * t4)
X4 <- matrix(scalar, nrow = 1)
n_train4 <- 400

hresult <- hankel_dmd(X4[, 1:n_train4, drop = FALSE], delays = 30, rank = 4, dt = dt4)
hrecon <- hankel_reconstruct(hresult, n_train4)
hpred <- predict(hresult, n_ahead = n4 - n_train4)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
plot(t4[1:n_train4], scalar[1:n_train4], type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", main = "Hankel-DMD Reconstruction (Training)")
lines(t4[1:n_train4], hrecon[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Reconstructed"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2), bg = "white")

plot(t4, scalar, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", main = "Hankel-DMD Out-of-Sample Prediction")
lines(t4[(n_train4 + 1):n4], hpred[1, ], col = col_predict, lwd = 2, lty = 2)
abline(v = t4[n_train4], lty = 3, col = "gray50")
legend("topright", c("Actual", "Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2), bg = "white")
mtext("Hankel-DMD: Two-Frequency Scalar Signal", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 6: GLA -- Reconstruction and Prediction
# =============================================================================
png(file.path(outdir, "gla_prediction.png"), width = 900, height = 600, res = 120)
n5 <- 400; t5 <- seq(0, (n5 - 1) * 0.1, by = 0.1)
gla_signal <- rbind(sin(t5), cos(t5))
gresult <- gla(gla_signal, n_eigenvalues = 2, tol = 1e-4)
grecon <- gla_reconstruct(gresult)
gpred <- predict(gresult, n_ahead = 100)
t_future <- seq(t5[n5] + 0.1, by = 0.1, length.out = 100)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
plot(t5, gla_signal[1, ], type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "x1", main = "GLA Reconstruction")
lines(t5, grecon[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "GLA Reconstructed"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2), bg = "white")

plot(t_future, sin(t_future), type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "x1", main = "GLA Out-of-Sample Prediction")
lines(t_future, gpred[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("True Future", "GLA Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2), bg = "white")
mtext("Generalized Laplace Analysis", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 7: Phase Portraits
# =============================================================================
png(file.path(outdir, "dynamical_maps.png"), width = 900, height = 600, res = 120)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

traj_reg <- generate_trajectory("standard", c(0.1, 0.2), 5000, epsilon = 0.12)
plot(traj_reg[1, ], traj_reg[2, ], pch = ".", col = col_actual, cex = 0.5,
     xlab = "x", ylab = "y", main = "Standard Map (eps=0.12)")

traj_chaos <- generate_trajectory("standard", c(0.5, 0.5), 50000, epsilon = 0.97)
plot(traj_chaos[1, ], traj_chaos[2, ], pch = ".", col = col_predict, cex = 0.3,
     xlab = "x", ylab = "y", main = "Standard Map (eps=0.97)")

traj_henon <- generate_trajectory("henon", c(0.0, 0.0), 20000, a = 1.4, b = 0.3)
plot(traj_henon[1, ], traj_henon[2, ], pch = ".", col = col_var2, cex = 0.3,
     xlab = "x", ylab = "y", main = "Henon Attractor")

traj_log <- generate_trajectory("logistic", 0.4, 200, r = 3.9)
plot(1:201, traj_log[1, ], type = "l", col = col_recon, lwd = 1,
     xlab = "Iteration", ylab = "x", main = "Logistic Map (r=3.9)")
mtext("Built-in Dynamical Maps", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 8: HTA Convergence
# =============================================================================
png(file.path(outdir, "hta_convergence.png"), width = 900, height = 400, res = 120)
conv_reg <- hta_convergence("standard", c(0.1, 0.2), observable = "sin_pi",
                             omega = 0.0, n_iter = 50000, epsilon = 0.12)
conv_chaos <- hta_convergence("standard", c(0.5, 0.5), observable = "sin_pi",
                               omega = 0.0, n_iter = 50000, epsilon = 0.97)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
plot(conv_reg$times, conv_reg$hta_magnitudes, type = "l",
     col = col_actual, lwd = 2, log = "x",
     xlab = "Iterations", ylab = "|HTA|", main = "Regular Orbit (eps=0.12)")
plot(conv_chaos$times, conv_chaos$hta_magnitudes, type = "l",
     col = col_predict, lwd = 2, log = "x",
     xlab = "Iterations", ylab = "|HTA|", main = "Chaotic Orbit (eps=0.97)")
mtext("HTA Convergence: Regular vs Chaotic", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 9: Mesochronic Plot
# =============================================================================
png(file.path(outdir, "mesochronic_plot.png"), width = 900, height = 400, res = 120)
meso <- mesochronic_compute("standard", x_range = c(0, 1), y_range = c(0, 1),
                             resolution = 100, observable = "sin_pi",
                             omega = 0.0, n_iter = 5000, epsilon = 0.12)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 2), oma = c(0, 0, 2, 0))
image(meso$x_coords, meso$y_coords, meso$hta_matrix,
      col = hcl.colors(256, "viridis"), xlab = "x", ylab = "y", main = "|HTA| Magnitude")
image(meso$x_coords, meso$y_coords, meso$phase_matrix,
      col = hcl.colors(256, "RdYlBu"), xlab = "x", ylab = "y", main = "HTA Phase")
mtext("Mesochronic Harmonic Plot (Standard Map, eps=0.12)", outer = TRUE, cex = 1.1, font = 2)
dev.off()

# =============================================================================
# Plot 10: Method Comparison
# =============================================================================
png(file.path(outdir, "method_comparison.png"), width = 900, height = 700, res = 120)
n10 <- 300; dt10 <- 0.01; t10 <- seq(0, (n10 - 1) * dt10, by = dt10)
test_sig <- sin(2 * pi * 2.0 * t10) + 0.3 * sin(2 * pi * 7.0 * t10)
X10 <- matrix(test_sig, nrow = 1)
n_train10 <- 250; n_test10 <- n10 - n_train10
X10_train <- X10[, 1:n_train10, drop = FALSE]
actual_test <- test_sig[(n_train10 + 1):n10]
t_test10 <- t10[(n_train10 + 1):n10]

dmd_r <- dmd(X10_train, rank = 4, dt = dt10)
dmd_p <- predict(dmd_r, n_ahead = n_test10)
hdmd_r <- hankel_dmd(X10_train, delays = 30, rank = 4, dt = dt10)
hdmd_p <- predict(hdmd_r, n_ahead = n_test10)
gla_r <- gla(X10_train, n_eigenvalues = 4, tol = 1e-4)
gla_p <- predict(gla_r, n_ahead = n_test10)

err_d <- sqrt(mean((dmd_p[1, ] - actual_test)^2))
err_h <- sqrt(mean((hdmd_p[1, ] - actual_test)^2))
err_g <- sqrt(mean((gla_p[1, ] - actual_test)^2))
ylim10 <- range(c(actual_test, dmd_p[1, ], hdmd_p[1, ], gla_p[1, ]))

par(mfrow = c(3, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
plot(t_test10, actual_test, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", ylim = ylim10,
     main = sprintf("Standard DMD (RMSE = %.4f)", err_d))
lines(t_test10, dmd_p[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("Actual", "DMD"), col = c(col_actual, col_predict),
       lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)

plot(t_test10, actual_test, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", ylim = ylim10,
     main = sprintf("Hankel-DMD (RMSE = %.4f)", err_h))
lines(t_test10, hdmd_p[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Hankel-DMD"), col = c(col_actual, col_recon),
       lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)

plot(t_test10, actual_test, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", ylim = ylim10,
     main = sprintf("GLA (RMSE = %.4f)", err_g))
lines(t_test10, gla_p[1, ], col = col_var2, lwd = 2, lty = 2)
legend("topright", c("Actual", "GLA"), col = c(col_actual, col_var2),
       lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)
mtext("Method Comparison: Out-of-Sample Prediction", outer = TRUE, cex = 1.1, font = 2)
dev.off()

cat("All plots generated in", outdir, "\n")
