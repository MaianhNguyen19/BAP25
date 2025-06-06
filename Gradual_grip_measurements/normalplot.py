import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, lfilter
import os

# === FIR Filter Function ===
def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    """
    Design a Blackman‐window FIR bandpass filter that passes frequencies
    between lowcut and highcut (in Hz) and apply it to `data`.
    """
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
    """
    Compute a moving RMS over `signal` with a sliding window of length
    `window_size_samples`. Returns an array of RMS values, one per window.
    """
    rms_values = []
    for i in range(len(signal) - window_size_samples + 1):
        window = signal[i : i + window_size_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    return np.array(rms_values)


# === Settings ===
fs = 1000  # Sampling rate of your oscilloscope data (Hz)
window_ms = 500
window_size = int(fs * (window_ms / 1000))  # 1000 ms window → fs * 1 second

csv_path = "sannegrip.csv"  # path to your CSV file

# === 1) Load and clean the CSV ===
df = pd.read_csv(csv_path)

# Drop any rows where osc_ch1 or osc_ch2 are NaN
osc = df.dropna(subset=["osc_ch1", "osc_ch2"]).copy()

# Convert timestamp to float (if not already) and zero‐reference
osc["timestamp"] = osc["timestamp"].astype(float)
osc["timestamp"] -= osc["timestamp"].iloc[0]

# === 2) Normalize both channels (min–max to [0,1]) ===
osc["osc_ch1_norm"] = (
    osc["osc_ch1"] - osc["osc_ch1"].min()
) / (osc["osc_ch1"].max() - osc["osc_ch1"].min())

osc["osc_ch2_norm"] = (
    osc["osc_ch2"] - osc["osc_ch2"].min()
) / (osc["osc_ch2"].max() - osc["osc_ch2"].min())

# === 3) Convert Channel 2 to actual voltage ===
# In your original plotting, you multiplied osc_ch2 by 30 to get volts.
# So we do the same here before filtering.
osc["osc_ch2_volts"] = osc["osc_ch2"] * 30.0  # now in volts

# === 4) Apply 50 Hz bandpass filter to Channel 2 voltage ===
lowcut = 45.0   # Hz
highcut = 55.0  # Hz
numtaps = 200  # filter length (adjust for steeper roll‐off if needed)

filtered_ch2 = blackman_bandpass_filter(
    osc["osc_ch2_volts"].values, lowcut, highcut, fs, numtaps=numtaps
)

# === 5) Compute moving RMS of the filtered Channel 2 ===
rms_ch2 = moving_rms(filtered_ch2, window_size)

# Optionally drop the first (window_size − 1) samples from the time axis:
time_rms = osc["timestamp"].iloc[: len(rms_ch2)].values

# === 6) Plot results ===
plt.figure(figsize=(14, 8))

# 6a) Plot normalized oscilloscope channels
plt.subplot(2, 1, 1)
plt.plot(
    osc["timestamp"],
    osc["osc_ch1_norm"],
    label="Osc CH1 (normalized)",
    linewidth=1,
)
plt.plot(
    osc["timestamp"],
    osc["osc_ch2_norm"],
    label="Osc CH2 (normalized)",
    linewidth=1,
)
plt.title("Normalized Oscilloscope Channels")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)

# 6b) Plot filtered CH2 and its RMS
plt.subplot(2, 1, 2)
plt.plot(
    osc["timestamp"],
    osc["osc_ch2_volts"],
    color="gray",
    alpha=0.4,
    label="Osc CH2 (raw, in volts ×30)",
)
plt.plot(
    osc["timestamp"],
    osc["osc_ch1_norm"],
    color="blue",
    alpha=0.7,
    label="Osc CH2 (50 Hz bandpass filtered)",
)
plt.plot(
    time_rms,
    rms_ch2*10/3,  # convert RMS from V → mV if you prefer mV units
    color="red",
    linewidth=2,
    label=f"Moving RMS (window={window_ms} ms)",
)
plt.title("Channel 2: Filtered Signal and Moving RMS")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V) / RMS (mV)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
