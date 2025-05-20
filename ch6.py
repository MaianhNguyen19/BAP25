import serial
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import re

# === CONFIGURATION ===
PORT     = 'com3'
BAUD     = 921600
DURATION = 3        # seconds
CSV_FILE = 'mcp3208_6ch.csv'

# === OPEN & FLUSH SERIAL PORT ===
ser = serial.Serial(PORT, BAUD, timeout=1)
ser.reset_input_buffer()   # drop any junk left over
print(f"Connected to {PORT} @ {BAUD} baud")

start_time = time.time()
data = []

print("Recording... (Ctrl-C to stop early)")
try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        # skip blank or non-CSV lines
        if not line or not re.match(r'^\d+,', line):
            continue

        parts = line.split(',')
        if len(parts) != 7:
            continue

        try:
            ts    = int(parts[0])
            chans = list(map(int, parts[1:]))
        except ValueError:
            continue

        data.append([ts, *chans])

        if DURATION and (time.time() - start_time) > DURATION:
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    ser.close()
    print(f"Serial closed, collected {len(data)} samples")

if not data:
    raise RuntimeError("No valid data received!  Check that your ESP32 is printing ASCII CSV.")

# === SAVE TO CSV ===
cols = ['timestamp_us'] + [f'ch{i}' for i in range(6)]
df = pd.DataFrame(data, columns=cols)
df['time_s'] = (df['timestamp_us'] - df['timestamp_us'].iloc[0]) / 1e6
df.to_csv(CSV_FILE, index=False)
print(f"Saved {len(df)} samples to {CSV_FILE}")

# === PLOT ===
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(df['time_s'], df[f'ch{i}'], label=f'ch{i}',)
plt.xlabel('Time (s)')
plt.ylabel('ADC Value')
plt.title('MCP3208 6‐Channel Acquisition')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
