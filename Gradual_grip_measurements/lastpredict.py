import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# === 1) Calibration constants for CH1 → Weight → Force ===
#    (From your 6‐point calibration: R_sensor = a · w^b)
weights_calib     = np.array([128, 256, 259, 518, (2*128 + 518), 1000])  # grams
resistances_calib = np.array([20e3, 10e3,  9.1e3, 5e3,   2.7e3,       2.1e3])  # Ω

# Fit R_sensor = a · w^b via log–log linear regression
log_w = np.log(weights_calib)
log_R = np.log(resistances_calib)
b, log_a = np.polyfit(log_w, log_R, 1)
a = np.exp(log_a)
print(f"Calibration: R_sensor(w) = {a:.3e}·w^{b:.3f}   (Ω, w in g)")

R_FIXED = 10e3   # 10 kΩ fixed resistor in the divider
V_SUPPLY = 3.3   # 3.3 V supply

def resistance_from_voltage_ch1(V_ch1):
    """
    Invert the voltage divider on Channel 1:
    V_ch1 = 3.3 · (10k / (10k + R_sensor))  →  R_sensor
    """
    return R_FIXED * (V_SUPPLY / V_ch1 - 1)

def weight_from_voltage_ch1(V_ch1):
    """
    Given the measured divider voltage V_ch1 (0 < V_ch1 < 3.3 V),
    return the corresponding weight (g) from the power‐law calibration.
    """
    R_pred = resistance_from_voltage_ch1(V_ch1)
    return (R_pred / a) ** (1.0 / b)

def force_from_voltage_ch1(V_ch1):
    """
    Chain: V_ch1 → weight (g) → force (N).
    (1 g = 0.001 kg, multiply by 9.81 m/s²)
    """
    w_pred = weight_from_voltage_ch1(V_ch1)
    return (w_pred / 1000.0) * 9.81  # convert g → kg → N


# === 2) Load CSV and clean the data ===
csv_path = "sannegrip.csv"   # ← replace with your actual filename
df = pd.read_csv(csv_path)

# Drop any rows where osc_ch1 or osc_ch2 is NaN
osc = df.dropna(subset=["osc_ch1", "osc_ch2"]).copy()

# Convert timestamp to float and zero‐reference
osc["timestamp"] = osc["timestamp"].astype(float)
osc["timestamp"] -= osc["timestamp"].iloc[0]

# Convert raw ADC to volts
# (Your original scale: CH1 × 10 → V, CH2 × 30 → V)
osc["ch1_volts"] = osc["osc_ch1"] * 10.0
osc["ch2_volts"] = osc["osc_ch2"] * 30.0

# Discard any rows where CH1_volts ≤ 0 or ≥ 3.3 (invalid for divider inversion)
mask_valid = (osc["ch1_volts"] > 0) & (osc["ch1_volts"] < V_SUPPLY)
osc = osc[mask_valid].reset_index(drop=True)


# === 3) Compute Force vs. Time from Channel 1 ===
#    (Produces a full-length time series of force in newtons)
osc["force_N"] = osc["ch1_volts"].apply(force_from_voltage_ch1)


# === 4) Bandpass‐filter Channel 2 (45–55 Hz) and compute moving RMS ===
fs = 1000        # sampling rate in Hz
lowcut  = 45.0   # Hz
highcut = 55.0   # Hz
numtaps = 200    # FIR filter length

def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps):
    """
    Design a Blackman‐window FIR bandpass filter passing [lowcut, highcut],
    then apply lfilter to the data.
    """
    taps = firwin(
        numtaps,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window="blackman"
    )
    return lfilter(taps, 1.0, data)

def moving_rms(signal, window_size):
    """
    Compute sliding‐window RMS over 'signal' with 'window_size' samples.
    Returns an array of length len(signal) − window_size + 1.
    """
    rms_vals = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i : i + window_size]
        rms_vals.append(np.sqrt(np.mean(window**2)))
    return np.array(rms_vals)

# 4a) Filter CH2 around 50 Hz
filtered_ch2 = blackman_bandpass_filter(
    osc["ch2_volts"].values, lowcut, highcut, fs, numtaps
)

# 4b) Use a 100 ms window (window_ms) for RMS
window_ms = 100  # milliseconds
window_size = int(fs * (window_ms / 1000.0))  # e.g., 100 samples
if len(filtered_ch2) < window_size:
    raise RuntimeError(f"Recording has only {len(filtered_ch2)} samples, which is shorter than a {window_ms} ms window.")

rms_ch2 = moving_rms(filtered_ch2, window_size)

# 4c) Build a time axis for the RMS values:
#     Center each window at its midpoint: half_win seconds after the window start
half_win = (window_size / 2.0) / fs  # in seconds
time_rms = osc["timestamp"].iloc[: len(rms_ch2)].values + half_win

# 4d) Trim the force array to match RMS length
force_trimmed = osc["force_N"].iloc[: len(rms_ch2)].values


# === 5) Scatter and fit: linear–linear & log–log, power‐law vs. log‐linear ===

# 5a) Quick scatter plots
plt.figure(figsize=(10, 4))

# 5a.1) Linear–linear scatter
plt.subplot(1, 2, 1)
plt.scatter(rms_ch2, force_trimmed, s=20, color="black")
plt.xlabel("CH2 50 Hz RMS (V)")
plt.ylabel("Force (N)")
plt.title("Scatter: Force vs. CH2_RMS (linear–linear)")
plt.grid(True)

# 5a.2) Log–log scatter (only positive values)
mask_pos = (rms_ch2 > 0) & (force_trimmed > 0)
plt.subplot(1, 2, 2)
plt.scatter(rms_ch2[mask_pos], force_trimmed[mask_pos], s=20, color="blue")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("CH2 50 Hz RMS (V) [log]")
plt.ylabel("Force (N) [log]")
plt.title("Scatter: Force vs. CH2_RMS (log–log)")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()


# === 5b) Fit a power‐law: Force ≈ A * (RMS)^B ===
x = rms_ch2[mask_pos]
y = force_trimmed[mask_pos]
log_x = np.log(x)
log_y = np.log(y)

# Linear regression in log–log space: log(y) ≈ log(A) + B * log(x)
B, logA = np.polyfit(log_x, log_y, 1)
A = np.exp(logA)
print(f"Power‐law fit:  Force ≈ {A:.3e} * RMS^{B:.3f}")

# Compute R² in log–log space
y_pred_log = logA + B * log_x
ss_res_log = np.sum((log_y - y_pred_log) ** 2)
ss_tot_log = np.sum((log_y - np.mean(log_y)) ** 2)
r2_loglog = 1 - (ss_res_log / ss_tot_log)
print(f"  → R² (log–log) = {r2_loglog:.4f}")

# For plotting the power‐law curve:
x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
y_fit_power = A * (x_fit ** B)


# === 5c) Fit a log‐linear model: Force ≈ α * ln(RMS) + β ===
α, β = np.polyfit(log_x, y, 1)
print(f"Log‐linear fit:  Force ≈ {α:.3e} * ln(RMS) + {β:.3e}")

# Compute R² for log‐linear on (x,y)
y_pred_lin = α * log_x + β
ss_res_lin = np.sum((y - y_pred_lin) ** 2)
ss_tot_lin = np.sum((y - np.mean(y)) ** 2)
r2_linlog = 1 - (ss_res_lin / ss_tot_lin)
print(f"  → R² (lin vs. ln‐x) = {r2_linlog:.4f}")

# For plotting the log‐linear curve:
y_fit_loglin = α * np.log(x_fit) + β


# === 6) Plot data + both fits on log–log axes ===
plt.figure(figsize=(6, 5))
plt.scatter(rms_ch2, force_trimmed, color="black", s=20, label="Data")

# Power‐law curve (red)
plt.plot(x_fit, y_fit_power, color="red", lw=2,
         label=f"Power‐law: y={A:.2e}·x^{B:.2f}")

# Log‐linear curve (green)
plt.plot(x_fit, y_fit_loglin, color="green", lw=2,
         label=f"Log‐linear: y={α:.2e}·ln(x)+{β:.2e}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("CH2 50 Hz RMS (V) [log scale]")
plt.ylabel("Force (N) [log scale]")
plt.title("Comparison: Power‐law vs. Log‐linear Fits")
plt.legend(fontsize="small")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
