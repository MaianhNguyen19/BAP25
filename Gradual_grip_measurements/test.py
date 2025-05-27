import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\steph\OneDrive\Documents\BAP25\RecordedData\merged_esp_scope.csv")


# 1) Examine sample intervals
dts = np.diff(df["Time_s"].dropna())
print(f"ESP32 Δt: mean={np.mean(dts):.6f}s, std={np.std(dts):.6f}s")

# 2) Convert ADC counts to volts
df["adc6_v"] = df["adc6"] * (5.0/4096)

# 3) Scatter-plot all three
plt.figure(figsize=(10,5))
plt.scatter(df["Time_s"], df["adc6_v"], s=5, label="ESP32 CH6", c="C0")
plt.plot(  df["Time_s"], df["ch1"],  label="Scope CH1", alpha=0.8)
plt.plot(  df["Time_s"], df["ch2"],  label="Scope CH2", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Sampling Verification")
plt.legend()
plt.grid()
plt.show()
