import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the merged CSV
csv_path = r"C:\Users\steph\OneDrive\Documents\BAP25\RecordedData\digilent_ch1_ch2.csv"
df = pd.read_csv(csv_path)

# 2. Plot
plt.figure(figsize=(12, 6))
plt.plot(df["Time_s"], df["adc6"], label="ESP32 ADC Ch6", linewidth=1)
plt.plot(df["Time_s"], df["ch1"],  label="Scope CH1",      linewidth=1, alpha=0.8)
plt.plot(df["Time_s"], df["ch2"],  label="Scope CH2",      linewidth=1, alpha=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Voltage / ADC Code")
plt.title("Combined ESP32 & Scope Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
