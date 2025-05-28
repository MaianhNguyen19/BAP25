import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("esp32_test_maianhsleep.csv")
plt.figure(figsize=(8,3))
plt.plot(df["Time_s"][:1000], df["ADC6"][:1000], '-o', markersize=2)
plt.xlabel("Time (s)")
plt.ylabel("ADC6 Counts")
plt.title("ESP32 Raw Sampling")
plt.grid(True)
plt.show()
