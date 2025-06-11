import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from scipy.stats import pearsonr

# === 0) Parameters ===
fs = 1000            # Sampling rate (Hz)
drop_ms = 200        # Drop first 200 ms to avoid filter transients
drop_samples = int(fs * drop_ms / 1000)

# === 1) Calibration data (CH1) ===
weights_calib     = np.array([128, 256, 259, 518, (2*128 + 518), 1000])  # g
resistances_calib = np.array([20e3, 10e3, 9.1e3, 5e3, 2.7e3, 2.1e3])   # Ω

# Fit R = a·w^b
log_w = np.log(weights_calib)
log_R = np.log(resistances_calib)
b, log_a = np.polyfit(log_w, log_R, 1)
a = np.exp(log_a)

# Divider constants
R_FIXED  = 2.7e3   # Ω
V_SUPPLY = 3.3     # V

def weight_from_voltage_ch1(V):
    R_s = R_FIXED * (V_SUPPLY / V - 1)
    return (R_s / a)**(1.0 / b)

# === 2) Load & clean ===
csv_path = "daivdlgrip.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["osc_ch1", "osc_ch2"]).copy()
df["timestamp"] = df["timestamp"].astype(float)
df["timestamp"] -= df["timestamp"].iloc[0]

# Convert to volts
df["ch1_volts"] = df["osc_ch1"] * 10.0
df["ch2_volts"] = df["osc_ch2"] * 30.0
df = df[(df["ch1_volts"] > 0) & (df["ch1_volts"] < V_SUPPLY)].reset_index(drop=True)

# Drop startup samples in df
df = df.iloc[drop_samples:].reset_index(drop=True)

# === 3) CH1 smoothing & weight ===
taps1 = firwin(101, 5.0, fs=fs)
ch1_fwd = lfilter(taps1, 1.0, df["ch1_volts"].values)
df["ch1_smoothed"] = lfilter(taps1, 1.0, ch1_fwd[::-1])[::-1]
df["weight_g"]     = df["ch1_smoothed"].apply(weight_from_voltage_ch1)

# === 4) CH2 50 Hz bandpass & fast RMS ===
taps2 = firwin(200, [45,55], pass_zero=False, fs=fs, window="blackman")
ch2_fwd = lfilter(taps2, 1.0, df["ch2_volts"].values)
ch2_bp  = lfilter(taps2, 1.0, ch2_fwd[::-1])[::-1]

window_fast = int(0.005 * fs)
def moving_rms(x, N):
    return np.sqrt(np.convolve(x**2, np.ones(N)/N, mode="valid"))

rms_fast = moving_rms(ch2_bp, window_fast)

# now that we know rms_fast length, sync time & weight
time_rms    = df["timestamp"].iloc[:len(rms_fast)].values
weight_sync = df["weight_g"].iloc[:len(rms_fast)].values

# drop startup portion from RMS, time, and weight arrays
rms_fast    = rms_fast[drop_samples:]
time_rms    = time_rms[drop_samples:]
weight_sync = weight_sync[drop_samples:]

# === 5) EMA smoothing of RMS ===
tau   = 0.2
alpha = 1 - np.exp(-1/(tau*fs))
rms_slow = np.zeros_like(rms_fast)
rms_slow[0] = rms_fast[0]
for i in range(1, len(rms_fast)):
    rms_slow[i] = alpha * rms_fast[i] + (1-alpha) * rms_slow[i-1]

# === 6) Baseline subtract & shift ===
rms_min      = rms_slow.min()
rms_shifted  = rms_slow - rms_min
rms_shifted[rms_shifted < 0] = 0

# === 7) Fit power-law: weight = A·rms_shifted^B ===
mask = (rms_shifted > 0) & (weight_sync > 0)
log_x = np.log(rms_shifted[mask])
log_y = np.log(weight_sync[mask])
B, logA = np.polyfit(log_x, log_y, 1)
A = np.exp(logA)

def weight_from_rms(val):
    v = val - rms_min
    return 0.0 if v <= 0 else A * (v**B)

# === 8) Correlation & summary ===
r_val, p_val = pearsonr(rms_shifted[mask], weight_sync[mask])
print(f"Power‐law fit: weight = {A:.3e}·RMS^{B:.3f}")
print(f"Pearson r = {r_val:.3f}, p = {p_val:.2e}")

# === 9) Plot ===
plt.figure(figsize=(12,9))

plt.subplot(3,1,1)
plt.plot(df["timestamp"], df["ch1_volts"], alpha=0.4, label="CH1 Raw")
plt.plot(df["timestamp"], df["ch1_smoothed"], label="CH1 Smoothed")
plt.ylabel("Voltage (V)"); plt.legend(); plt.grid()

plt.subplot(3,1,2)
plt.plot(time_rms, rms_slow/2.5, label="CH2 RMS Slow")
plt.plot(df["timestamp"], df["ch1_smoothed"]/2.3, label="CH1 Smoothed")
plt.ylabel("RMS (V)"); plt.legend(); plt.grid()

plt.subplot(3,1,3)
plt.plot(df["timestamp"], df["osc_ch2"]*30, label="Oscilloscope Channel 2", linewidth=1)
plt.plot(df["timestamp"], df["ch1_volts"], alpha=0.4, label="CH1 Raw")
plt.xlabel("Shifted RMS (V)"); plt.ylabel("Weight (g)")
plt.legend(); plt.grid(which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, find_peaks

# Bandpass‐filter CH2 around 50 Hz (reuse your taps2 design):
taps2 = firwin(200, [45, 55], pass_zero=False, fs=fs, window="blackman")
ch2_fwd = lfilter(taps2, 1.0, df["ch2_volts"].values)
ch2_bp  = lfilter(taps2, 1.0, ch2_fwd[::-1])[::-1]

# 1) Detect peaks in the absolute CH2 waveform:
peaks_idx, _ = find_peaks(np.abs(ch2_bp), height=0)

# 2) Get peak times and values:
peak_times = df["timestamp"].values[peaks_idx]
peak_vals  = ch2_bp[peaks_idx]

# 3) Interpolate CH1‐derived weight at those peak times:
weight_interp = np.interp(peak_times, df["timestamp"], df["weight_g"])

# 4) Plot weight time‐series with CH2 peaks marked:
plt.figure(figsize=(8,4))
plt.plot(df["timestamp"], df["weight_g"]/1000, label="CH1 Weight (g)")
#plt.scatter(peak_times, weight_interp, color='C1', s=10, label="CH2 Peak Events")
plt.scatter(peak_times, peak_vals, color='C1', s=10, label="CH2 Peak Events")
plt.xlabel("Time (s)")
plt.ylabel("Weight (g)")
plt.title("Grip Weight vs. CH2 Peak Events")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.signal import find_peaks
window_fast= 10
# 1) Compute a moving-peak envelope over the same window size as RMS:
#    (window_fast is your RMS window in samples)
def moving_peak(x, N):
    return np.array([np.max(np.abs(x[i:i+N])) for i in range(len(x) - N + 1)])

# Make sure ch2_bp and window_fast are already defined
peaks_env = moving_peak(ch2_bp, window_fast)

# Build a time vector aligned to peak windows:
time_peaks = df["timestamp"].values[:len(peaks_env)]

# 2) Trim the same startup samples you dropped earlier:
peaks_env = peaks_env[drop_samples:]
time_peaks = time_peaks[drop_samples:]

# 3) Plot both features together:
plt.figure(figsize=(10, 4))
plt.plot(time_rms, rms_slow,      label="RMS (slow EMA)", linewidth=1)
plt.plot(time_peaks, peaks_env,   label="Peak envelope",  linewidth=1, alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("CH2 50 Hz Feature Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
from scipy.signal import lfilter, firwin
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# assume `bp` is your band-passed CH2, and weight_true is time-aligned CH1 weight

# 1) Rectify (absolute value)
rect = np.abs(df["ch2_volts"])

# 2) Smooth with the same 5 Hz low-pass you used before
lp   = firwin(101, 5.0, fs=1000)
env2 = lfilter(lp, 1, rect)
env2 = lfilter(lp, 1, env2[::-1])[::-1]

# 3) Align by lag_samples and trim
env2_a = np.roll(env2, -lag_samples)[:-lag_samples]
w_a    = weight_true[lag_samples:lag_samples+len(env2_a)]

# 4) Correlation metrics
r, p      = pearsonr(env2_a, w_a)
r2        = r**2
rmse      = np.sqrt(mean_squared_error(w_a, env2_a))
norm_rmse = rmse / (w_a.max() - w_a.min()) * 100

print(f"Rectified+LP Pearson r = {r:.3f}, R² = {r2:.3f}, Norm. RMSE = {norm_rmse:.1f}%")