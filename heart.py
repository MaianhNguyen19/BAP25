import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

def notch_filter(data, notch_freq=50.0, fs=2000.0, Q=30):
    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, data)

# === Load your recorded CSV ===
df = pd.read_csv("mcp3208_6ch.csv").iloc[1:]
df['timestamp_s'] = df['timestamp_us'] * 1e-6
fs = 2000  # Hz

# === Extract raw heart signal (e.g. channel 0) ===
signal = np.array(df['ch0'], dtype=np.float32)
signal -= np.mean(signal)
signal = notch_filter(signal)

# === Apply a simple moving average (denoise)
def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

smoothed = smooth(signal, window_size=5)



# === Plot
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp_s'], smoothed, label='Smoothed Heart Signal')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
