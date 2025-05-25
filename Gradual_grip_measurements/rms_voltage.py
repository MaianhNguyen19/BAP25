import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, butter, lfilter

# === Filter Design Functions ===

def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs, window='blackman')
    return lfilter(taps, 1.0, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return lfilter(b, a, data)

# === Settings ===
fs = 1000  # Sampling rate
lowcut = 0.0001
highcut = 499.0
base_path = "/Users/davidlacle/Documents/TUDelft/BAP/BAP25/Gradual_grip_measurements/"
file_suffixes = ["1m", "3f", "4f", "5m"]
base_name = "mcp3208_6ch_measurement_grip"

for suffix in file_suffixes:
    filename = f"{base_path}{base_name}{suffix}_attempt2.csv"
    df = pd.read_csv(filename).iloc[1:]  # Drop first row

    # === Prepare signal ===
    signal = np.array(df['ch0'], dtype=np.float32) * 5 / 4096
    N = len(signal)
    time = np.array(df['time_s'], dtype=np.float32)

    # === Apply Filters ===
    filtered_fir = blackman_bandpass_filter(signal, lowcut, highcut, fs)
    filtered_iir = butter_bandpass_filter(signal, lowcut, highcut, fs)

    # === FFT Calculation ===
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_fir = 20 * np.log10(np.abs(np.fft.fft(filtered_fir)) + 1e-12)
    fft_iir = 20 * np.log10(np.abs(np.fft.fft(filtered_iir)) + 1e-12)

    # === Plot Frequency Domain Comparison ===
    plt.figure(figsize=(12, 5))
    plt.plot(freqs[:N//2], fft_fir[:N//2], label="FIR (Blackman)")
    plt.plot(freqs[:N//2], fft_iir[:N//2], label="Butterworth (IIR)")
    plt.title(f"FFT Comparison - {suffix}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Optional: Plot Time Domain ===
    plt.figure(figsize=(12, 4))
    plt.plot(time, filtered_fir, label="FIR Filtered", alpha=0.7)
    plt.plot(time, filtered_iir, label="IIR Filtered", alpha=0.7)
    plt.title(f"Time Domain Signal - {suffix}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()