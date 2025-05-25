import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, butter, freqz

# === Settings ===
fs = 1000  # Sampling frequency in Hz
lowcut = 40
highcut = 50

# === FIR Filter (Blackman Window) ===
numtaps = 500
fir_taps = firwin(
    numtaps=numtaps,
    cutoff=[lowcut, highcut],
    window='blackman',
    pass_zero=False,
    fs=fs
)
w_fir, h_fir = freqz(fir_taps, worN=2048, fs=fs)

# === IIR Filter (Butterworth) ===
order = 4
b_iir, a_iir = butter(order, [lowcut, highcut], btype='band', fs=fs)
w_iir, h_iir = freqz(b_iir, a_iir, worN=2048, fs=fs)

# === Plot Magnitude (dB) ===
plt.figure(figsize=(12, 5))
plt.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-12), label='FIR (Blackman)')
plt.plot(w_iir, 20 * np.log10(np.abs(h_iir) + 1e-12), label='Butterworth (IIR)')
plt.title("Frequency Response (Magnitude in dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# === Plot Phase Response ===
plt.figure(figsize=(12, 5))
plt.plot(w_fir, np.angle(h_fir, deg=True), label='FIR (Blackman)')
plt.plot(w_iir, np.angle(h_iir, deg=True), label='Butterworth (IIR)')
plt.title("Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (degrees)")
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()