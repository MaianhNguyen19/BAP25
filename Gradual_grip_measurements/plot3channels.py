import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("sannegrip.csv")

# Drop rows with missing oscilloscope data
osc = df.dropna(subset=["osc_ch1", "osc_ch2"])
esp = df.dropna(subset=["esp_value"])


# Plot both channels
plt.figure(figsize=(12, 5))

plt.plot(osc["timestamp"], osc["osc_ch1"]*10, label="Oscilloscope Channel 1", linewidth=1)
plt.plot(osc["timestamp"], osc["osc_ch2"]*30, label="Oscilloscope Channel 2", linewidth=1)
# Plot ESP32 signal
esp["timestamp"]+2*1000
plt.plot(esp["timestamp"], (esp["esp_value"]*5)/4096, 'o-', label="ESP32 Sensor", markersize=3, alpha=0.8)

plt.title("Oscilloscope 2-Channel Signal")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
