import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# 1) Load your CSV (timestamps in sec, adc counts in 'adc6')
csv_path = r"C:\Users\steph\OneDrive\Documents\BAP25\RecordedData\merged_esp_scope.csv"
df = pd.read_csv(csv_path)

# Use only the ESP32 column
t = df["Time_s"].values
adc = df["adc6"].values

# 2) (Optional) convert ADC counts to voltage; here assuming 0–5V over 12 bits:
voltage = adc * (5.0/4096)

# 3) Band-pass filter design: 0.5–5 Hz (heart rates ~30–300 bpm)
fs = 1/np.mean(np.diff(t))  # actual sampling rate
lowcut, highcut = 0.5, 15.0
b, a = butter(N=3, Wn=[lowcut/(fs/2), highcut/(fs/2)], btype='band')
ecg_filt = filtfilt(b, a, voltage)

# 4) R-peak detection: find peaks above a dynamic threshold
#    set minimal distance between peaks to 0.4s (max 150 bpm)
min_dist = int(0.4 * fs)
peaks, _ = find_peaks(ecg_filt, distance=min_dist, height=np.std(ecg_filt)*0.5)

# 5) Instantaneous heart rate (bpm)
peak_times = t[peaks]
rr_intervals = np.diff(peak_times)            # seconds between beats
hr_inst = 60.0 / rr_intervals                  # bpm
hr_times = peak_times[1:]                      # assign each HR to the second peak

# Smooth HR with a rolling window
hr_series = pd.Series(hr_inst, index=hr_times).rolling(5, min_periods=1).mean()

# 6) Plot everything
plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)
plt.plot(t, ecg_filt, 'k', label='Filtered ECG (0.5–5 Hz)')
plt.plot(t[peaks], ecg_filt[peaks], 'ro', label='R-peaks')
plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
plt.legend(); plt.grid(True)
plt.title("ECG-like Signal & Detected R-Peaks")

plt.subplot(2,1,2)
plt.plot(hr_series.index, hr_series.values, 'b-')
plt.xlabel("Time (s)"); plt.ylabel("Heart Rate (bpm)")
plt.grid(True)
plt.title("Instantaneous Heart Rate")

plt.tight_layout()
plt.show()
