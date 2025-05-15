import serial
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# Change as needed
measurement_time_sec = 3
signal_frequency = 150
attempt_nr = 5

# Replace with your ESP32's serial port.
#921600
port = 'COM3'
baud = '921600'

filename = "humantenna"
filename_1 = f"measurement_{signal_frequency}Hz_{measurement_time_sec}s_attempt{attempt_nr}.csv"
path = r"C:\Users\harim\Desktop\grip_code\BAP25\Measurement code\Measurements"
foldername = "Measurements"
file_path = os.path.join(path, filename)

data =[]
count = 0
samples_for_1_sec = math.ceil(1/ 370e-6)
amount_of_samples = measurement_time_sec * samples_for_1_sec

# Open serial port
ser = serial.Serial(port, baud)
print("Connected to", port)

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp_us', 'grip_voltage','sample_interval_us'])

    print("Logging data...")

    try:
        while count<= amount_of_samples:
            line = ser.readline().decode('utf-8', errors="ignore").strip()
            print(line)
            

            if ',' in line:
                try:
                    timestamp_us, grip = line.split(',')
                    voltage = float(grip) * 3.3/4095
                    data.append({'timestamp_us': int(timestamp_us), 'grip_voltage': float(voltage)})
                    count += 1
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
for row in data:
    row['relative_time_seconds'] = row['relative_time_us'] * 10**-6

df = pd.DataFrame(data)

df.to_csv(filename, index=False)
print(f"Saved {len(df)} samples to {filename}")