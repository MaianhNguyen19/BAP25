import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, lfilter, get_window
import os

# === FIR Filter Function (Blackman Bandpass) ===
def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(
        numtaps,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window='blackman'
    )
    return lfilter(taps, 1.0, data)

# === Sliding RMS with Arbitrary Window + 50% Overlap ===
def sliding_rms(signal, time, window, hop):
    window_size = len(window)
    output = []
    time_out = []
    for start in range(0, len(signal) - window_size + 1, hop):
        segment = signal[start:start + window_size]
        rms_val = np.sqrt(np.sum(segment**2 * window))
        output.append(rms_val)
        time_out.append(time[start + window_size // 2])  # center-aligned
    return np.array(time_out), np.array(output)

# === FFT Function ===
def compute_fft_db(signal, fs):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / n
    fft_db = 20 * np.log10(fft_mag + 1e-12)  # avoid log(0)
    return freq, fft_db

# === Settings ===
fs = 1000  # Sampling rate in Hz
window_ms = 20*3  # window duration in milliseconds
window_size = int(fs * (window_ms / 1000))  # convert to samples
hop_size = window_size // 2
base_path = "C:/Users/steph/OneDrive/Documents/BAP25/Gradual_grip_measurements_nicelines/"
file_suffixes = ["1f", "1m", "2f", "2m", "3f", "3m"]
base_name = "mcp3208_6ch_measurement_grip"
all_stats = []

plt.figure(figsize=(12, 6))

for suffix in file_suffixes:
    filename = os.path.join(base_path, f"{base_name}{suffix}_attempt3.csv")
    df = pd.read_csv(filename).iloc[2:]  # Drop first 2 header rows if needed
    df["time_s"] = df["time_s"].astype(float)
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]

    # === Load & scale signal ===
    signal = np.array(df['ch0'], dtype=np.float32) * 5 / 4096
    time = df['time_s'].values

    # === Remove first 500 samples and reset time ===
    aq=10
    signal = signal[aq:]
    time = time[aq:]
    time = time - time[0]

    # === Apply Bandpass Filter (45–55 Hz) ===
    filtered = blackman_bandpass_filter(signal, 45, 55, fs, numtaps=200)

    # === Define Rectangular and Blackman Windows ===
    window_rect = np.ones(window_size)
    window_rect /= np.sum(window_rect)

    window_black = get_window('blackman', window_size)
    window_black /= np.sum(window_black)


    # === Compute RMS with 50% Overlap ===
    time_rect, rms_rect = sliding_rms(filtered, time, window_rect, hop_size)
    time_black, rms_black = sliding_rms(filtered, time, window_black, hop_size)

    # === Remove the first 500 RMS samples (due to filter ringing) ===
    a=20
    rms_rect = rms_rect[a:]
    time_rect = time_rect[a:]
    rms_black = rms_black[a:]
    time_black = time_black[a:]

    # === Normalize to Zero & Convert to mV ===
    rms_rect_zeroed = (rms_rect - rms_rect[0]) * 1000
    rms_black_zeroed = (rms_black - rms_black[0]) * 1000

    # === Store Stats ===
    stats = {
        "file": suffix,
        "mean_rect": np.mean(rms_rect_zeroed),
        "std_rect": np.std(rms_rect_zeroed),
        "mean_black": np.mean(rms_black_zeroed),
        "std_black": np.std(rms_black_zeroed)
    }
    all_stats.append(stats)

    # === Plot RMS Curves ===
    plt.plot(time_rect, rms_rect_zeroed, label=f'Rectangular RMS - {suffix}')
    plt.plot(time_black, rms_black_zeroed, label=f'Blackman RMS - {suffix}', linestyle='--')

# === Final Plot Settings ===
plt.title("RMS with 50% Overlap (Rectangular vs. Blackman) for Filtered Grip Signal")
plt.xlabel("Time (s)")
plt.ylabel("RMS Voltage (mV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print RMS Summary Table ===
stats_df = pd.DataFrame(all_stats)
print("📊 RMS Summary Statistics (after baseline normalization, in mV):")
print(stats_df.to_string(index=False))
