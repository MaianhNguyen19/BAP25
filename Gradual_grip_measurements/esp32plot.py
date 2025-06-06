import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("merged.csv")
plt.figure(figsize=(8,3))
plt.plot(df["Time_s"], df["CH0"], '-o', markersize=2)
#plt.plot(df["Time_s"], df["ADC6"], '-o', markersize=2)
plt.xlabel("Time (s)")
plt.ylabel("ADC6 Counts")
plt.title("ESP32 Raw Sampling")
plt.grid(True)
plt.show()
