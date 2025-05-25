import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, lfilter

# === FIR Filter Functions ===
def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(
        numtaps,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window='blackman'
    )
    return lfilter(taps, 1.0, data)
# === RMS Function ===
def moving_rms(signal, window_size_samples):
    rms_values = []
    for i in range(len(signal) - window_size_samples + 1):
        window = signal[i:i + window_size_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    return np.array(rms_values)

# === FFT Function ===
def compute_fft_db(signal, fs):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / n
    fft_db = 20 * np.log10(fft_mag + 1e-12)  # avoid log(0)
    return freq, fft_db


# === Settings ===
fs = 1000  # Sampling rate
window_ms = 1000
window_size = int(fs * (window_ms / 1000))
base_path = "/Users/davidlacle/Documents/TUDelft/BAP/BAP25/Gradual_grip_measurements_nicelines/"
file_suffixes = ["1f","1m", "2f", "2m","3f","3m"]
#file_suffixes = ["1m","2f", "3f", "4f","5m","6m"]
base_name = "mcp3208_6ch_measurement_grip"
#base_name = "sine_50hz_4Vpp"
all_stats = []
plt.figure(figsize=(12, 6))

for suffix in file_suffixes:
    filename = f"{base_path}{base_name}{suffix}{"_attempt3"}.csv"
    #filename = f"{base_path}{base_name}.csv"
    df = pd.read_csv(filename).iloc[2:]  # Drop first row
    df["time_s"] = df["time_s"].astype(float)
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]

    # Scale ADC to voltage
    signal = np.array(df['ch0'], dtype=np.float32) * 5 / 4096

    # Filter signal (50 Hz band)
    filtered = blackman_bandpass_filter(signal, 45 ,55, fs, numtaps=200)

    # Compute RMS
    rms = moving_rms(filtered, window_size)
    rms = rms[10:]  # just remove the first few points if needed               # remove first 10 RMS points just to clean it up
    rms_zeroed = rms
    rms_zeroed = rms_zeroed*1000
    # Time axis for RMS
    time_trimmed = df['time_s'].iloc[:len(rms)]
    # time_trimmed = time_s

    # Compute statistics
    stats = {
        "file": suffix,
        "mean": np.mean(rms_zeroed),
        "std": np.std(rms_zeroed),
        "min": np.min(rms_zeroed),
        "max": np.max(rms_zeroed),
        "range": np.max(rms_zeroed) - np.min(rms_zeroed)
    }
    all_stats.append(stats)



    # Plot RMS
    #plt.plot(df["time_s"], filtered, label="Normalized RMS")
    plt.plot(time_trimmed, rms_zeroed, label=f"RMS - {suffix}")

# === Final Plot Settings ===
plt.title("RMS of Filtered Signal (45–55 Hz) Across 6 Measurements")
plt.xlabel("Time (s)")
plt.ylabel("RMS Voltage (mV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Compute FFTs ===
freq_raw, fft_raw_db = compute_fft_db(signal, fs)
freq_filt, fft_filt_db = compute_fft_db(filtered, fs)

# === Plot Comparison ===
plt.figure(figsize=(10, 5))
plt.plot(freq_raw, fft_raw_db, label="Raw Signal")
plt.plot(freq_filt, fft_filt_db, label="Filtered Signal (45–55 Hz)")
plt.title("FFT Before and After Filtering (dB Scale)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-100, 0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Table of Stats ===
stats_df = pd.DataFrame(all_stats)
print("📊 RMS Statistics (after ringing & baseline correction):")
print(stats_df.to_string(index=False))