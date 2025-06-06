import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("experiment_log.csv")

# Prepare oscilloscope data (host timestamp)
#osc = df.dropna(subset=["osc_value"])

# Prepare ESP32 data using host timestamp (ignore esp_timestamp)
esp = df.dropna(subset=["esp_value"])

# Plot both
plt.figure(figsize=(14, 5))

# Plot oscilloscope signal
# #plt.plot(osc["timestamp"], osc["osc_value"]*10,'o-', label="Oscilloscope", alpha=0.7)

# Plot ESP32 signal
plt.plot(esp["timestamp"], (esp["esp_value"]*5)/4096, 'o-', label="ESP32 Sensor", markersize=3, alpha=0.8)

plt.xlabel("Host Timestamp (seconds)")
plt.ylabel("Signal Value")
plt.title("Oscilloscope and ESP32 Sensor Signals Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
