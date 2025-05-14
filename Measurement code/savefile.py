import serial
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

import serial.tools.list_ports
import time

# Replace with your ESP32's serial port
port = 'COM3'
baud = '921600'
filename = "grip_data_1.csv"
data =[]


# Open serial port
ser = serial.Serial(port, baud)
print("Connected to", port)

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp_us', 'grip_voltage','sample_interval_us'])

    print("Logging data...")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors="ignore").strip()
            print(line)

            if ',' in line:
                try:
                    timestamp_us, grip = line.split(',')
                    data.append({'timestamp_us': int(timestamp_us), 'grip_voltage': int(grip)})
                except ValueError:
                    print("Invalid line:", line)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        ser.close()
        print("Serial connection closed.")

starttime = data[0]['timestamp_us']
print(starttime)
for row in data:
    row['relative_time_us'] = row['timestamp_us'] - starttime

df = pd.DataFrame(data)

df.to_csv(filename, index=False)
print(f"Saved {len(df)} samples to {filename}")

#print(df)
# starttime = df['timestamp_us'].iloc[0]
# print("Start time:", starttime)
# df['relative_time_us'] = df['timestamp_us'] - starttime
# print(df)
# df.to_csv('grip_data_with_relative_time.csv', index=False)

# df = pd.read_csv('grip_data_1.csv')
# df = df.head(50)
# # Plot grip vs time
# plt.figure(figsize=(10, 5))
# plt.plot(df['timestamp_us'], df['grip_voltage'], marker='o', linestyle='-')
# plt.xlabel('Time (microseconds)')
# plt.ylabel('Grip Strength')
# plt.title('Grip Strength Over First 50 Samples')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
