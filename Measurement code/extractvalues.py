import numpy as np
import matplotlib.pyplot as plt

def moving_rms(signal, window_size_samples):
    rms_values = []
    for i in range(len(signal) - window_size_samples + 1):
        window = signal[i:i+window_size_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    return np.array(rms_values)
# Parameters
fs = 1000        # Sampling frequency (Hz)
f = 50           # Signal frequency (Hz)
T = 0.5            # Duration in seconds
t = np.linspace(0, T, int(T * fs), endpoint=False)  # Time vector

# Generate sine wave
signal = (2/t)*np.sin(2 * np.pi * f * t)
size = 3
window_ms = 100
window_size = int(fs * (window_ms / 1000))
rms = moving_rms(signal, window_size)

# Adjust time axis for RMS
t_rms = t[:len(rms)]

plt.plot(t, signal, label='Signal')
plt.plot(t_rms, rms, label='RMS', linewidth=2)
plt.legend()
plt.title('Sine Wave and Moving RMS')
plt.grid(True)
plt.show()