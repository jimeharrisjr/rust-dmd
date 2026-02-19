#!/usr/bin/env Rscript
# =============================================================================
# koopman.dmd -- Demonstration Script
#
# This script demonstrates the key features of the koopman.dmd R package
# with plots comparing predicted vs actual values.
# =============================================================================

library(koopman.dmd)

# Save all plots to PDF
pdf("koopman_dmd_demo.pdf", width = 11, height = 8.5)

# Color palette
col_actual  <- "#2196F3"   # blue
col_predict <- "#FF5722"   # red-orange
col_recon   <- "#4CAF50"   # green
col_var2    <- "#9C27B0"   # purple
col_grid    <- "#E0E0E040"

# =============================================================================
# 1. CORE DMD -- Oscillatory System
# =============================================================================

cat("=== 1. Core DMD: Two-Variable Oscillation ===\n")

dt <- 0.05
n_train <- 200
n_total <- 300
t_all <- seq(0, (n_total - 1) * dt, by = dt)
t_train <- t_all[1:n_train]
t_test  <- t_all[(n_train + 1):n_total]

# Two coupled oscillators with slight damping
x1 <- exp(-0.05 * t_all) * sin(2 * pi * 0.5 * t_all)
x2 <- exp(-0.05 * t_all) * cos(2 * pi * 0.5 * t_all)
X_all <- rbind(x1, x2)
X_train <- X_all[, 1:n_train]
X_test  <- X_all[, (n_train + 1):n_total]

# Fit DMD
result <- dmd(X_train, rank = 2, dt = dt)
cat("DMD Result:\n")
summary(result)

# Predict future (out-of-sample)
n_ahead <- n_total - 1  # predict from t=1 onward
pred <- predict(result, n_ahead = n_ahead)

# -- Plot 1: Predicted vs Actual (both variables) --
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

plot(t_all, x1, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "x1",
     main = "Variable 1: Predicted vs Actual")
lines(t_all[2:n_total], pred[1, ], col = col_predict, lwd = 2, lty = 2)
abline(v = t_train[n_train], lty = 3, col = "gray50")
text(t_train[n_train], max(x1) * 0.9, "Train | Test",
     pos = 4, cex = 0.8, col = "gray40")
legend("topright", c("Actual", "DMD Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

plot(t_all, x2, type = "l", col = col_var2, lwd = 2,
     xlab = "Time", ylab = "x2",
     main = "Variable 2: Predicted vs Actual")
lines(t_all[2:n_total], pred[2, ], col = col_predict, lwd = 2, lty = 2)
abline(v = t_train[n_train], lty = 3, col = "gray50")
legend("topright", c("Actual", "DMD Predicted"),
       col = c(col_var2, col_predict), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

mtext("Core DMD: Damped Oscillator", outer = TRUE, cex = 1.2, font = 2)

# -- Plot 2: Reconstruction Error --
recon <- dmd_reconstruct(result)
err <- dmd_error(result)
cat(sprintf("Reconstruction RMSE: %.6f\n", err$rmse))
cat(sprintf("Relative Error:      %.6f\n", err$relative_error))

par(mfrow = c(1, 1), mar = c(4, 4, 3, 1))
residuals_x1 <- X_train[1, ] - recon[1, ]
residuals_x2 <- X_train[2, ] - recon[2, ]

plot(t_train, residuals_x1, type = "l", col = col_actual, lwd = 1.5,
     xlab = "Time", ylab = "Residual",
     main = sprintf("DMD Reconstruction Residuals (RMSE = %.2e)", err$rmse),
     ylim = range(c(residuals_x1, residuals_x2)))
lines(t_train, residuals_x2, col = col_var2, lwd = 1.5)
abline(h = 0, lty = 2, col = "gray50")
legend("topright", c("x1 residual", "x2 residual"),
       col = c(col_actual, col_var2), lwd = 1.5, bg = "white", cex = 0.8)

# -- Plot 3: Eigenvalue Spectrum --
spec <- dmd_spectrum(result)
cat("\nSpectrum:\n")
print(spec)

stab <- dmd_stability(result)
cat(sprintf("\nStability: spectral_radius=%.4f, stable=%s\n",
            stab$spectral_radius, stab$is_stable))

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Unit circle with eigenvalues
theta <- seq(0, 2 * pi, length.out = 200)
plot(cos(theta), sin(theta), type = "l", col = "gray70", lwd = 1,
     xlab = "Re", ylab = "Im", asp = 1,
     main = "Eigenvalues in Complex Plane",
     xlim = c(-1.3, 1.3), ylim = c(-1.3, 1.3))
abline(h = 0, v = 0, col = "gray85")
points(result$eigenvalues_re, result$eigenvalues_im,
       pch = 19, col = col_predict, cex = 1.5)

# Frequency spectrum
barplot(spec$magnitude, names.arg = round(spec$frequency, 3),
        col = col_actual, border = NA,
        xlab = "Frequency (Hz)", ylab = "|lambda|",
        main = "Mode Magnitudes")
abline(h = 1.0, lty = 2, col = "gray50")

mtext("DMD Spectral Analysis", outer = TRUE, line = -1, cex = 1.1, font = 2)


# =============================================================================
# 2. DMD WITH CENTERING -- Multi-Frequency Signal with Mean
# =============================================================================

cat("\n=== 2. DMD with Centering ===\n")

n <- 300
t2 <- seq(0, (n - 1) * 0.02, by = 0.02)
signal <- 3.0 + 2.0 * sin(2 * pi * 1.0 * t2) + 0.5 * sin(2 * pi * 3.0 * t2)
X2 <- matrix(signal, nrow = 1)

# Without centering
res_no_center <- dmd(X2, rank = 4, dt = 0.02)
pred_no_center <- predict(res_no_center, n_ahead = n - 1)

# With centering
res_center <- dmd(X2, rank = 4, center = TRUE, dt = 0.02)
pred_center <- predict(res_center, n_ahead = n - 1)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

plot(t2, signal, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal",
     main = "Without Centering")
lines(t2[2:n], pred_no_center[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("Actual", "Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

plot(t2, signal, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal",
     main = "With Centering")
lines(t2[2:n], pred_center[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Predicted (centered)"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

mtext("Effect of Mean Centering on DMD", outer = TRUE, cex = 1.2, font = 2)


# =============================================================================
# 3. EXTENDED DMD -- Polynomial Lifting for Nonlinear Signal
# =============================================================================

cat("\n=== 3. Extended DMD: Polynomial Lifting ===\n")

n3 <- 200
t3 <- seq(0, (n3 - 1) * 0.05, by = 0.05)
nonlinear_signal <- sin(t3)^2  # nonlinear
X3 <- matrix(nonlinear_signal, nrow = 1)

# Standard DMD
res_std <- dmd(X3, dt = 0.05)
pred_std <- predict(res_std, n_ahead = n3 - 1)
err_std <- dmd_error(res_std)

# Extended DMD with polynomial lifting
res_lift <- dmd(X3, lifting = "polynomial", lifting_param = 2, dt = 0.05)
pred_lift <- predict(res_lift, n_ahead = n3 - 1)
err_lift <- dmd_error(res_lift)

cat(sprintf("Standard DMD RMSE: %.6f\n", err_std$rmse))
cat(sprintf("Extended DMD RMSE: %.6f\n", err_lift$rmse))

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

plot(t3, nonlinear_signal, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "sin^2(t)",
     main = sprintf("Standard DMD (RMSE = %.4f)", err_std$rmse))
lines(t3[2:n3], pred_std[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("Actual", "Standard DMD"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

plot(t3, nonlinear_signal, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "sin^2(t)",
     main = sprintf("Extended DMD with Polynomial Lifting (RMSE = %.4f)", err_lift$rmse))
lines(t3[2:n3], pred_lift[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Polynomial Lifted DMD"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

mtext("Extended DMD: Lifting Improves Nonlinear Fit", outer = TRUE, cex = 1.2, font = 2)


# =============================================================================
# 4. HANKEL-DMD -- Scalar Time Series Frequency Recovery
# =============================================================================

cat("\n=== 4. Hankel-DMD: Scalar Time Series ===\n")

n4 <- 500
dt4 <- 0.02
t4 <- seq(0, (n4 - 1) * dt4, by = dt4)
freq1 <- 1.0
freq2 <- 3.0
scalar_signal <- sin(2 * pi * freq1 * t4) + 0.4 * sin(2 * pi * freq2 * t4)
X4 <- matrix(scalar_signal, nrow = 1)

n_train4 <- 400
X4_train <- X4[, 1:n_train4, drop = FALSE]

hresult <- hankel_dmd(X4_train, delays = 30, rank = 4, dt = dt4)
cat("Hankel-DMD Result:\n")
print(hresult)

# Reconstruct training portion
hrecon <- hankel_reconstruct(hresult, n_train4)

# Predict future
n_pred4 <- n4 - n_train4
hpred <- predict(hresult, n_ahead = n_pred4)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

# Reconstruction
plot(t4[1:n_train4], scalar_signal[1:n_train4], type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal",
     main = "Hankel-DMD Reconstruction (Training Data)")
lines(t4[1:n_train4], hrecon[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Hankel-DMD Reconstructed"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

# Prediction
plot(t4, scalar_signal, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal",
     main = "Hankel-DMD Out-of-Sample Prediction")
lines(t4[(n_train4 + 1):n4], hpred[1, ], col = col_predict, lwd = 2, lty = 2)
abline(v = t4[n_train4], lty = 3, col = "gray50")
text(t4[n_train4], max(scalar_signal) * 0.9, "Train | Test",
     pos = 4, cex = 0.8, col = "gray40")
legend("topright", c("Actual", "Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

mtext("Hankel-DMD: Scalar Time Series with Two Frequencies",
      outer = TRUE, cex = 1.2, font = 2)

# Eigenvalue analysis
eig_mag <- sqrt(hresult$eigenvalues_re^2 + hresult$eigenvalues_im^2)
eig_freq <- abs(atan2(hresult$eigenvalues_im, hresult$eigenvalues_re)) / (2 * pi * dt4)
cat(sprintf("Recovered frequencies: %s Hz\n",
            paste(round(eig_freq, 2), collapse = ", ")))
cat(sprintf("Eigenvalue magnitudes: %s\n",
            paste(round(eig_mag, 4), collapse = ", ")))


# =============================================================================
# 5. GLA -- Generalized Laplace Analysis
# =============================================================================

cat("\n=== 5. GLA: Generalized Laplace Analysis ===\n")

n5 <- 400
t5 <- seq(0, (n5 - 1) * 0.1, by = 0.1)
gla_signal <- rbind(sin(t5), cos(t5))

gresult <- gla(gla_signal, n_eigenvalues = 2, tol = 1e-4)
cat("GLA Result:\n")
print(gresult)

grecon <- gla_reconstruct(gresult)
gpred <- predict(gresult, n_ahead = 100)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

# Reconstruction
plot(t5, gla_signal[1, ], type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "x1",
     main = "GLA Reconstruction")
lines(t5, grecon[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "GLA Reconstructed"),
       col = c(col_actual, col_recon), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

# Prediction (extrapolation)
t_future <- seq(t5[n5] + 0.1, by = 0.1, length.out = 100)
true_future <- sin(t_future)
plot(t_future, true_future, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "x1",
     main = "GLA Out-of-Sample Prediction")
lines(t_future, gpred[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("True Future", "GLA Predicted"),
       col = c(col_actual, col_predict), lwd = 2, lty = c(1, 2),
       bg = "white", cex = 0.8)

mtext("Generalized Laplace Analysis", outer = TRUE, cex = 1.2, font = 2)


# =============================================================================
# 6. DYNAMICAL MAPS -- Phase Space Portraits
# =============================================================================

cat("\n=== 6. Dynamical Maps: Phase Portraits ===\n")

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

# Standard map -- regular orbit
traj_reg <- generate_trajectory("standard", c(0.1, 0.2), 5000, epsilon = 0.12)
plot(traj_reg[1, ], traj_reg[2, ], pch = ".", col = col_actual, cex = 0.5,
     xlab = "x", ylab = "y", main = "Standard Map (Regular, eps=0.12)")

# Standard map -- chaotic orbit
traj_chaos <- generate_trajectory("standard", c(0.5, 0.5), 50000, epsilon = 0.97)
plot(traj_chaos[1, ], traj_chaos[2, ], pch = ".", col = col_predict, cex = 0.3,
     xlab = "x", ylab = "y", main = "Standard Map (Chaotic, eps=0.97)")

# Henon attractor
traj_henon <- generate_trajectory("henon", c(0.0, 0.0), 20000, a = 1.4, b = 0.3)
plot(traj_henon[1, ], traj_henon[2, ], pch = ".", col = col_var2, cex = 0.3,
     xlab = "x", ylab = "y", main = "Henon Attractor (a=1.4, b=0.3)")

# Logistic map -- time series
traj_log <- generate_trajectory("logistic", 0.4, 200, r = 3.9)
plot(1:201, traj_log[1, ], type = "l", col = col_recon, lwd = 1,
     xlab = "Iteration", ylab = "x", main = "Logistic Map (r=3.9)")

mtext("Built-in Dynamical Maps", outer = TRUE, cex = 1.2, font = 2)


# =============================================================================
# 7. DMD ON MAP TRAJECTORY -- Learn Dynamics from Data
# =============================================================================

cat("\n=== 7. DMD on Standard Map Trajectory ===\n")

n_map <- 2000
traj <- generate_trajectory("standard", c(0.1, 0.2), n_map, epsilon = 0.12)

n_map_train <- 1500
X_map_train <- traj[, 1:n_map_train]
X_map_test  <- traj[, (n_map_train + 1):(n_map + 1)]

map_result <- dmd(X_map_train, rank = 4, dt = 1.0)
map_pred <- predict(map_result, n_ahead = n_map - n_map_train)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

plot(1:(n_map + 1), traj[1, ], type = "l", col = col_actual, lwd = 1.5,
     xlab = "Iteration", ylab = "x",
     main = "Standard Map x-coordinate: DMD Prediction")
lines((n_map_train + 1):(n_map + 1), c(X_map_test[1, 1], map_pred[1, ]),
      col = col_predict, lwd = 2, lty = 2)
abline(v = n_map_train, lty = 3, col = "gray50")
legend("topright", c("Actual", "DMD Predicted"),
       col = c(col_actual, col_predict), lwd = c(1.5, 2), lty = c(1, 2),
       bg = "white", cex = 0.8)

plot(1:(n_map + 1), traj[2, ], type = "l", col = col_var2, lwd = 1.5,
     xlab = "Iteration", ylab = "y",
     main = "Standard Map y-coordinate: DMD Prediction")
lines((n_map_train + 1):(n_map + 1), c(X_map_test[2, 1], map_pred[2, ]),
      col = col_predict, lwd = 2, lty = 2)
abline(v = n_map_train, lty = 3, col = "gray50")
legend("topright", c("Actual", "DMD Predicted"),
       col = c(col_var2, col_predict), lwd = c(1.5, 2), lty = c(1, 2),
       bg = "white", cex = 0.8)

mtext("DMD Applied to Standard Map Trajectory", outer = TRUE, cex = 1.2, font = 2)


# =============================================================================
# 8. HARMONIC TIME AVERAGES -- Phase Space Structure
# =============================================================================

cat("\n=== 8. Harmonic Time Averages ===\n")

# HTA convergence for regular vs chaotic orbits
conv_reg <- hta_convergence("standard", c(0.1, 0.2), observable = "sin_pi",
                             omega = 0.0, n_iter = 50000, epsilon = 0.12)
conv_chaos <- hta_convergence("standard", c(0.5, 0.5), observable = "sin_pi",
                               omega = 0.0, n_iter = 50000, epsilon = 0.97)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

plot(conv_reg$times, conv_reg$hta_magnitudes, type = "l",
     col = col_actual, lwd = 2, log = "x",
     xlab = "Iterations", ylab = "|HTA|",
     main = "Regular Orbit (eps=0.12)")

plot(conv_chaos$times, conv_chaos$hta_magnitudes, type = "l",
     col = col_predict, lwd = 2, log = "x",
     xlab = "Iterations", ylab = "|HTA|",
     main = "Chaotic Orbit (eps=0.97)")

mtext("HTA Convergence: Regular vs Chaotic", outer = TRUE, cex = 1.2, font = 2)


# =============================================================================
# 9. MESOCHRONIC HARMONIC PLOT
# =============================================================================

cat("\n=== 9. Mesochronic Harmonic Plot ===\n")

meso <- mesochronic_compute("standard",
                             x_range = c(0, 1), y_range = c(0, 1),
                             resolution = 80, observable = "sin_pi",
                             omega = 0.0, n_iter = 5000,
                             epsilon = 0.12)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 2), oma = c(0, 0, 2, 0))

# HTA magnitude
image(meso$x_coords, meso$y_coords, meso$hta_matrix,
      col = hcl.colors(256, "viridis"),
      xlab = "x", ylab = "y",
      main = "|HTA| Magnitude")

# HTA phase
image(meso$x_coords, meso$y_coords, meso$phase_matrix,
      col = hcl.colors(256, "RdYlBu"),
      xlab = "x", ylab = "y",
      main = "HTA Phase")

mtext("Mesochronic Harmonic Plot (Standard Map, eps=0.12, omega=0)",
      outer = TRUE, cex = 1.1, font = 2)


# =============================================================================
# 10. COMPARISON SUMMARY
# =============================================================================

cat("\n=== 10. Method Comparison ===\n")

# Create a challenging multi-frequency test signal
n10 <- 300
dt10 <- 0.01
t10 <- seq(0, (n10 - 1) * dt10, by = dt10)
test_signal <- sin(2 * pi * 2.0 * t10) + 0.3 * sin(2 * pi * 7.0 * t10)
X10 <- matrix(test_signal, nrow = 1)

n_train10 <- 250
n_test10 <- n10 - n_train10
X10_train <- X10[, 1:n_train10, drop = FALSE]

# Standard DMD
dmd_res <- dmd(X10_train, rank = 4, dt = dt10)
dmd_pred <- predict(dmd_res, n_ahead = n_test10)

# Hankel-DMD
hdmd_res <- hankel_dmd(X10_train, delays = 30, rank = 4, dt = dt10)
hdmd_pred <- predict(hdmd_res, n_ahead = n_test10)

# GLA
gla_res <- gla(X10_train, n_eigenvalues = 4, tol = 1e-4)
gla_pred <- predict(gla_res, n_ahead = n_test10)

# Compute prediction errors
actual_test <- test_signal[(n_train10 + 1):n10]
err_dmd  <- sqrt(mean((dmd_pred[1, ]  - actual_test)^2))
err_hdmd <- sqrt(mean((hdmd_pred[1, ] - actual_test)^2))
err_gla  <- sqrt(mean((gla_pred[1, ]  - actual_test)^2))

cat(sprintf("Standard DMD prediction RMSE:  %.6f\n", err_dmd))
cat(sprintf("Hankel-DMD prediction RMSE:    %.6f\n", err_hdmd))
cat(sprintf("GLA prediction RMSE:           %.6f\n", err_gla))

par(mfrow = c(3, 1), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
t_test10 <- t10[(n_train10 + 1):n10]
ylim10 <- range(c(actual_test, dmd_pred[1, ], hdmd_pred[1, ], gla_pred[1, ]))

plot(t_test10, actual_test, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", ylim = ylim10,
     main = sprintf("Standard DMD (RMSE = %.4f)", err_dmd))
lines(t_test10, dmd_pred[1, ], col = col_predict, lwd = 2, lty = 2)
legend("topright", c("Actual", "DMD"), col = c(col_actual, col_predict),
       lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)

plot(t_test10, actual_test, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", ylim = ylim10,
     main = sprintf("Hankel-DMD (RMSE = %.4f)", err_hdmd))
lines(t_test10, hdmd_pred[1, ], col = col_recon, lwd = 2, lty = 2)
legend("topright", c("Actual", "Hankel-DMD"), col = c(col_actual, col_recon),
       lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)

plot(t_test10, actual_test, type = "l", col = col_actual, lwd = 2,
     xlab = "Time", ylab = "Signal", ylim = ylim10,
     main = sprintf("GLA (RMSE = %.4f)", err_gla))
lines(t_test10, gla_pred[1, ], col = col_var2, lwd = 2, lty = 2)
legend("topright", c("Actual", "GLA"), col = c(col_actual, col_var2),
       lwd = 2, lty = c(1, 2), bg = "white", cex = 0.8)

mtext("Method Comparison: Out-of-Sample Prediction",
      outer = TRUE, cex = 1.2, font = 2)

# Close PDF
dev.off()

cat("\n=== All plots saved to koopman_dmd_demo.pdf ===\n")
cat("Demo complete.\n")
