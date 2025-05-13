import serial
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt



# Replace with your ESP32's serial port
port = 'COM3'
baud = 115200
filename = "grip_data_1.csv"
data =[]

# Open serial port
ser = serial.Serial(port, baud)
print("Connected to", port)

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp_ms', 'grip_strength','sample_interval_us'])

    print("Logging data...")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors="ignore").strip()
            print(line)

            if ',' in line:
                try:
                    timestamp, grip, sample_interval_us = line.split(',')
                    data.append({'timestamp_ms': int(timestamp), 'grip_strength': int(grip), 'sample_interval_us' : int(sample_interval_us)})
                except ValueError:
                    print("Invalid line:", line)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        ser.close()
        print("Serial connection closed.")

df = pd.DataFrame(data)
df.to_csv(filename, index=False)
print(f"Saved {len(df)} samples to {filename}")

df = pd.read_csv('grip_data_1.csv')

# df = df.head(50)

# # Plot grip vs time
# plt.figure(figsize=(10, 5))
# plt.plot(df['timestamp_ms'], df['grip_strength'], marker='o', linestyle='-')
# plt.xlabel('Time (ms)')
# plt.ylabel('Grip Strength')
# plt.title('Grip Strength Over First 50 Samples')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
