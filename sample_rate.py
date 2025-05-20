import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
# Change as needed
measurement_time_sec = 10
signal_frequency = 150
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

# Convert timestamp column (assumed in microseconds) to seconds
df['timestamp_s'] = df['timestamp_us'] * 1e-6

# Calculate time difference between samples
df = df.iloc[1:]
dt = np.diff(df['timestamp_s'])

avg_dt = np.mean(dt)
sampling_rate = 1 / avg_dt

# Statistics
min_dt = np.min(dt)
max_dt = np.max(dt)
jitter = np.std(dt)
print(jitter,avg_dt)
# Plot Δt to visualize variation
plt.figure(figsize=(10, 4))
plt.plot(dt * 1000)  # convert to ms
plt.title("Time between samples (ms)")
plt.xlabel("Sample index")
plt.ylabel("Δt (ms)")
plt.grid(True)
plt.tight_layout()
plt.show()

sampling_rate, avg_dt, min_dt, max_dt, jitter