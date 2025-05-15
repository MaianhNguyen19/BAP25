import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Change as needed
measurement_time_sec = 10
signal_frequency = 150
attempt_nr = 5

#samples_for_1_sec = math.ceil(1/ 370e-6)
#amount_of_samples = measurement_time_sec * samples_for_1_sec

#path = r"C:\Users\harim\Desktop\grip_code\BAP25\Measurement code\Measurements"
#foldername = "Measurements"
filename = "humantenna"
filename_1 = f"measurement_{signal_frequency}Hz_{measurement_time_sec}s_attempt{attempt_nr}.csv"
#file_path = os.path.join(path, filename)


df = pd.read_csv(filename)
#df = df.head(100)

# Plot grip vs time
plt.figure(figsize=(10, 5))
plt.plot(df['relative_time_seconds'], df['grip_voltage'], marker='o', linestyle='-')
plt.xlabel('Time (Seconds)')
plt.ylabel('Grip Strength')
plt.title(f"measurement_{signal_frequency}Hz_{measurement_time_sec}s_attempt{attempt_nr}")
plt.grid(True)
plt.tight_layout()
plt.show()