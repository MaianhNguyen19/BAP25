import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# === FIR Bandpass Filter ===
def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(
        numtaps,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window='blackman'
    )
    return lfilter(taps, 1.0, data), taps

# === Moving RMS ===
def moving_rms(signal, window_size_samples):
    return np.sqrt(np.convolve(signal**2, np.ones(window_size_samples)/window_size_samples, mode='valid'))

# === Settings ===
fs = 1000
lowcut = 5
highcut = 40
numtaps = 101
rms_window_ms = 200
rms_window = int(fs * rms_window_ms / 1000)
baseline_duration = 0.5  # seconds
baseline_samples = int(fs * baseline_duration)

base_path = "/Users/davidlacle/Documents/TUDelft/BAP/BAP25/Gradual_grip_measurements/"
base_name = "mcp3208_6ch_measurement_grip"
suffixes = ["1m", "3f", "4f", "5m", "2f"]
channel = "ch0"

# === Initialize stats collection ===
all_stats = []

plt.figure(figsize=(12, 6))

for suffix in suffixes:
    filename = f"{base_path}{base_name}{suffix}_attempt2.csv"
    df = pd.read_csv(filename).iloc[1:]  # skip header
    signal = df[channel].astype(float) * 5 / 4096  # convert to volts

    # Apply FIR filter and get filter taps
    filtered, taps = blackman_bandpass_filter(signal, lowcut, highcut, fs, numtaps)
    group_delay = (numtaps - 1) // 2

    # Remove ringing (group delay samples)
    filtered = filtered[group_delay:]

    # Compute moving RMS
    rms = moving_rms(filtered, rms_window)

    # Remove initial RMS window delay
    rms = rms[rms_window:]

    # Normalize to baseline
    baseline = np.mean(rms[:baseline_samples])
    rms_normalized = rms - baseline

    # Compute statistics
    stats = {
        "file": suffix,
        "mean": np.mean(rms_normalized),
        "std": np.std(rms_normalized),
        "min": np.min(rms_normalized),
        "max": np.max(rms_normalized),
        "range": np.max(rms_normalized) - np.min(rms_normalized)
    }
    all_stats.append(stats)

    # Plot each signal
    plt.plot(rms_normalized, label=suffix)

# === Final Plot ===
plt.title("Normalized RMS (Blackman Filtered, Ringing Removed)")
plt.xlabel("Sample Index")
plt.ylabel("RMS Voltage (normalized)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print Table of Stats ===
stats_df = pd.DataFrame(all_stats)
print("📊 RMS Statistics (after ringing & baseline correction):")
print(stats_df.to_string(index=False))