import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, lfilter
import os

# === FIR Filter Function ===
def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(
        numtaps,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window='blackman'
    )
    return lfilter(taps, 1.0, data)

# === Moving RMS Function ===
def moving_rms(signal, window_size_samples):
    rms_values = []
    for i in range(len(signal) - window_size_samples + 1):
        window = signal[i : i + window_size_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    return np.array(rms_values)

# === Settings ===
fs = 1000  # Sampling rate (Hz)
window_ms = 1000
window_size = int(fs * (window_ms / 1000))  # 1000 ms window

csv_path = "sannegrip.csv"  # path to your CSV file

# === 1) Load and clean the CSV ===
df = pd.read_csv(csv_path)
osc = df.dropna(subset=["osc_ch1", "osc_ch2"]).copy()
osc["timestamp"] = osc["timestamp"].astype(float)
osc["timestamp"] -= osc["timestamp"].iloc[0]

# === 2) Convert both channels to volts ===
osc["osc_ch1_volts"] = osc["osc_ch1"] * 10.0  # Channel 1 (as before)
osc["osc_ch2_volts"] = osc["osc_ch2"] * 30.0  # Channel 2 (as before)

# === 3) Filter Channel 2 around 50 Hz ===
lowcut = 45.0
highcut = 55.0
numtaps = 200
filtered_ch2 = blackman_bandpass_filter(
    osc["osc_ch2_volts"].values, lowcut, highcut, fs, numtaps=numtaps
)

# === 4) Compute moving RMS of filtered Channel 2 ===
rms_ch2 = moving_rms(filtered_ch2, window_size)
time_rms = osc["timestamp"].iloc[: len(rms_ch2)].values

# === 5) Trim Channel 1 to match RMS length ===
ch1_trim = osc["osc_ch1_volts"].iloc[: len(rms_ch2)].values
time_ch1_trim = osc["timestamp"].iloc[: len(rms_ch2)].values

# === 6) Min–max normalize both trimmed signals ===
def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

ch1_norm = min_max_norm(ch1_trim)
rms_ch2_norm = min_max_norm(rms_ch2)  # already in volts

# === 7) Plot normalized Channel 1 and normalized RMS of Channel 2 ===
plt.figure(figsize=(10, 5))
plt.plot(
    time_ch1_trim,
    ch1_norm,
    label="Channel 1 (normalized)",
    linewidth=1,
)
plt.plot(
    time_rms,
    rms_ch2_norm,
    label=f"Channel 2 50 Hz RMS (normalized, {window_ms} ms window)",
    color="red",
    linewidth=2,
)
plt.title("Normalized: Channel 1 vs. Channel 2 50 Hz RMS")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude (0→1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
