

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# === 1) CH1 → Weight calibration (power‐law) ===
weights_calib     = np.array([128, 256, 259, 518, (2*128 + 518), 1000])  # grams
resistances_calib = np.array([20e3, 10e3,  9.1e3, 5e3,   2.7e3,       2.1e3])  # Ω

# Fit R_sensor = a · w^b
log_w = np.log(weights_calib)
log_R = np.log(resistances_calib)
b, log_a = np.polyfit(log_w, log_R, 1)
a = np.exp(log_a)
print(f"Calibration: R_sensor(w) = {a:.3e}·w^{b:.3f}   (Ω, w in g)")

R_FIXED = 2.7e3   # Ω
V_SUPPLY = 3.3   # V

def resistance_from_voltage_ch1(V_ch1):
    """Invert divider: V_ch1 = 3.3·(10k/(10k+R_sensor)) → R_sensor."""
    return R_FIXED * (V_SUPPLY / V_ch1 - 1)

def weight_from_voltage_ch1(V_ch1):
    """Given divider voltage V_ch1, return predicted weight in grams."""
    R_pred = resistance_from_voltage_ch1(V_ch1)
    return (R_pred / a) ** (1.0 / b)  # w in grams


# === 2) Load CSV and clean ===
csv_path = "recpeter.csv"   # ← replace with your actual filename
df = pd.read_csv(csv_path)

# Drop rows where osc_ch1 or osc_ch2 is NaN
osc = df.dropna(subset=["osc_ch1", "osc_ch2"]).copy()

# Zero‐reference timestamp
osc["timestamp"] = osc["timestamp"].astype(float)
osc["timestamp"] -= osc["timestamp"].iloc[0]

# Convert raw ADC → volts (CH1×10, CH2×30)
osc["ch1_volts"] = osc["osc_ch1"] * 10.0
osc["ch2_volts"] = osc["osc_ch2"] * 30.0

# Keep only CH1_volts in (0, 3.3)
mask_valid = (osc["ch1_volts"] > 0) & (osc["ch1_volts"] < V_SUPPLY)
osc = osc[mask_valid].reset_index(drop=True)


# === 3) Smooth Channel 1 voltage to remove spikes ===
fs = 1000  # sampling rate in Hz

# Low-pass FIR for CH1 (cutoff 5 Hz)
numtaps_ch1 = 101
cutoff_ch1 = 5.0  # Hz
taps_ch1 = firwin(numtaps_ch1, cutoff_ch1, fs=fs, window="hamming")

# Apply zero‐phase filtering: forward then backward
ch1_smoothed = lfilter(taps_ch1, 1.0, osc["ch1_volts"].values)
ch1_smoothed = lfilter(taps_ch1, 1.0, ch1_smoothed[::-1])[::-1]

osc["ch1_smoothed"] = ch1_smoothed


# === 4) Compute Weight (g) vs. Time from Smoothed CH1 ===
osc["weight_g"] = osc["ch1_smoothed"].apply(weight_from_voltage_ch1)


# === 5) Bandpass-filter CH2 around 50 Hz and compute fast RMS ===
lowcut  = 30.0   # Hz
highcut = 70.0   # Hz
numtaps_ch2 = 200

def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps):
    taps = firwin(
        numtaps,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window="blackman"
    )
    return lfilter(taps, 1.0, data)

def moving_rms(signal, window_size):
    rms_vals = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i : i + window_size]
        rms_vals.append(np.sqrt(np.mean(window**2)))
    return np.array(rms_vals)

# Filter CH2
filtered_ch2 = blackman_bandpass_filter(
    osc["ch2_volts"].values, lowcut, highcut, fs, numtaps_ch2
)

# Fast RMS with 50 ms window (50 samples at 1 kHz)
window_ms_fast = 500
window_fast = int(fs * (window_ms_fast / 1000.0))
if len(filtered_ch2) < window_fast:
    raise RuntimeError(f"Recording too short ({len(filtered_ch2)} samples) for a 50 ms window.")

rms_ch2_fast = moving_rms(filtered_ch2, window_fast)
time_rms_fast = osc["timestamp"].iloc[: len(rms_ch2_fast)].values

# Trim weight array to match RMS length
weight_trimmed = osc["weight_g"].iloc[: len(rms_ch2_fast)].values


# === 6) Apply exponential moving average (EMA) to slow down RMS if needed ===
tau = 0.2  # seconds, adjust to match CH1 sensor’s own dynamics
alpha = 1 - np.exp(-1.0 / (tau * fs))

rms_ch2_slow = np.zeros_like(rms_ch2_fast)
rms_ch2_slow[0] = rms_ch2_fast[0]
for i in range(1, len(rms_ch2_fast)):
    rms_ch2_slow[i] = alpha * rms_ch2_fast[i] + (1 - alpha) * rms_ch2_slow[i - 1]


# === 7) Fit a power‐law: Weight ≈ A·(RMS_slow)^B ===
mask_pos = (rms_ch2_slow > 0) & (weight_trimmed > 0)
x = rms_ch2_slow[mask_pos]
y = weight_trimmed[mask_pos]
log_x = np.log(x)
log_y = np.log(y)

B, logA = np.polyfit(log_x, log_y, 1)
A = np.exp(logA)
print(f"Power‐law fit: Weight(g) ≈ {A:.3e}·(RMS_slow)^{B:.3f}")

# Build fit curve for plotting
x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
y_fit_power = A * (x_fit ** B)


# === 8) Plot everything ===
plt.figure(figsize=(10, 10))

# 8a) CH1 raw vs. smoothed voltage
plt.subplot(4, 1, 1)
plt.plot(osc["timestamp"], osc["ch1_volts"], color="gray", alpha=0.5, label="CH1 raw")
plt.plot(osc["timestamp"], osc["ch1_smoothed"], color="blue", label="CH1 smoothed")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Channel 1: Raw vs. Smoothed (5 Hz low-pass)")
plt.legend()
plt.grid(True)

# 8b) Weight (g) from CH1 smoothed vs. time
plt.subplot(4, 1, 2)
plt.plot(osc["timestamp"], osc["weight_g"], color="navy", lw=1)
plt.xlabel("Time (s)")
plt.ylabel("Weight (g)")
plt.title("Weight from CH1 Smoothed vs. Time")
plt.grid(True)

# 8c) CH2 RMS: fast (red) vs. slow (orange) vs. time
plt.subplot(4, 1, 3)
plt.plot(time_rms_fast, rms_ch2_fast, color="red", lw=1, label="RMS fast (50 ms)")
plt.plot(time_rms_fast, rms_ch2_slow, color="orange", lw=2, label=f"RMS slow (τ={tau}s)")
plt.xlabel("Time (s)")
plt.ylabel("RMS (V)")
plt.title("CH2 50 Hz RMS: Fast vs. EMA Smoothed")
plt.legend()
plt.grid(True)

# 8d) Scatter: (RMS_slow vs. Weight) + power-law fit on log–log axes
plt.subplot(4, 1, 4)
x = rms_ch2_slow
weight_trimmed = osc["ch1_smoothed"]
print(len(osc["ch1_smoothed"]))
n = min(len(x), len(weight_trimmed))
x = x[:n]
y = weight_trimmed[:n]
plt.scatter(x,y, color="black", s=20, label="Data")
#plt.plot(x_fit, y_fit_power, color="green", lw=2, label=f"Fit: w={A:.2e}·x^{B:.2f}")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("CH2 RMS_slow (V) [log scale]")
plt.ylabel("Weight (g) [log scale]")
plt.title("Correlation: Smoothed RMS → Weight (Power-Law Fit)")
plt.legend(fontsize="small")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()

# === 9) Compute Pearson correlation between CH2 RMS_slow and Weight ===
from scipy.stats import pearsonr

# Make sure we use the same mask as in the fitting step
valid = mask_pos  # defined in step 7: (rms_ch2_slow>0) & (weight_trimmed>0)
x_corr = rms_ch2_slow[valid]
y_corr = weight_trimmed[valid]

# Pearson’s r and two‐tailed p‐value
r_value, p_value = pearsonr(x_corr, y_corr)
print(f"Pearson correlation: r = {r_value:.3f}, p = {p_value:.3e}")


# --- after computing filtered_ch2 and rms_ch2_fast as before ---
# apply EMA smoothing
tau = 0.2
alpha = 1 - np.exp(-1/(tau * fs))
rms_ch2_slow = np.zeros_like(rms_ch2_fast)
rms_ch2_slow[0] = rms_ch2_fast[0]
for i in range(1, len(rms_ch2_fast)):
    rms_ch2_slow[i] = alpha * rms_ch2_fast[i] + (1-alpha) * rms_ch2_slow[i-1]

# now compute predicted weight/pressure
A, B =  a, b   # replace with your actual fit values
weight_pred = A * (rms_ch2_slow ** B)

# plot it
plt.figure(figsize=(8,4))
plt.plot(time_rms_fast, weight_pred, label='Predicted weight from CH2')
plt.xlabel('Time (s)')
plt.ylabel('Weight (g)')
plt.title('Estimate from Non-pressure Sensor (CH2)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# weight_trimmed is osc["weight_g"][:len(rms_ch2_slow)]
x = rms_ch2_slow
y = weight_trimmed
print(len(x))

plt.figure(figsize=(6,6))
plt.scatter(x, y, color="black", s=20)
plt.xlabel("CH2 RMS_slow (V)")
plt.ylabel("Weight from CH1 (g)")
plt.title("Scatter: CH2 RMS vs. Weight")
plt.grid(True)
plt.show()
