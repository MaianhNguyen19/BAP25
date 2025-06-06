import numpy as np

# (1) Your original data points
weights     = np.array([128, 256, 259, 518, (2*128 + 518), 1000])   # in grams
resistances = np.array([20e3, 10e3,  9.1e3, 5e3,   2.7e3,      2.1e3])  # in Ω

# (2) Re‐compute the power‐law fit parameters (a, b)
log_w = np.log(weights)
log_R = np.log(resistances)
b, log_a = np.polyfit(log_w, log_R, 1)
a = np.exp(log_a)

# (3) Predicted resistances on the original points
R_pred = a * (weights ** b)

# (4) Calculate residuals and error metrics
residuals = resistances - R_pred
N = len(resistances)

# RMSE
rmse = np.sqrt(np.mean(residuals**2))

# R^2
R_mean = np.mean(resistances)
ss_res = np.sum((resistances - R_pred)**2)
ss_tot = np.sum((resistances - R_mean)**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Power‐law fit: R_sensor(w) = {a:.3e}·w^{b:.3f}")
print(f"RMSE (Ω) = {rmse:.1f}")
print(f"R²        = {r_squared:.4f}")
