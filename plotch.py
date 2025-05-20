import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy.signal import firwin, lfilter

# Sampling rate and desired cutoff frequency (in Hz)
fs = 2000          # example sampling rate, adjust to yours
cutoff = 20        # example cutoff frequency (low-pass)

# Number of filter taps (the more, the sharper the cutoff)
numtaps = 101      # must be odd

# Design the FIR filter
fir_coeff = firwin(numtaps, cutoff, fs=fs)


# Change as needed
measurement_time_sec = 10
signal_frequency = 500
attempt_nr = 5


#samples_for_1_sec = math.ceil(1/ 370e-6)
#amount_of_samples = measurement_time_sec * samples_for_1_sec

#path = r"C:\Users\harim\Desktop\grip_code\BAP25\Measurement code\Measurements"
#foldername = "Measurements"
filename = "mcp3208_6ch.csv"
filename_1 = f"measurement_{signal_frequency}Hz_{measurement_time_sec}s_attempt{attempt_nr}.csv"
#file_path = os.path.join(path, filename)


df = pd.read_csv(filename)
#df = df.head(100)
df['timestamp_s'] = df['timestamp_us'] * 1e-6
df['ch4']
# Calculate time difference between samples
df = df.iloc[1:]
dt = np.diff(df['timestamp_s'])
signal = np.array(df['ch0'], dtype=np.float32)*3.3/4096
# Assuming `data` is a 1D NumPy array (e.g., from one ADC channel)
signal = lfilter(fir_coeff, 1.0, signal)

print(signal)


# === PLOT ===
plt.figure(figsize=(10, 6))
# for i in range(1):
#     plt.plot(df['timestamp_s'], df[f'ch{i}'],'-o', label=f'ch{i}',)
plt.plot(df['timestamp_s'], signal,'-', label=f'ch{0}')
plt.xlabel('Time (s)')
plt.ylabel('ADC Value')
plt.title('MCP3208 6‐Channel Acquisition')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
