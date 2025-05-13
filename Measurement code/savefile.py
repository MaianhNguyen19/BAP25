import serial
import csv
import time
import pandas as pd

# Replace with your ESP32's serial port
port = 'COM3'
baud = 115200
filename = "grip_data_1.csv"
data =[]
sample_limit = 10
sample_count = 0

# Open serial port
ser = serial.Serial(port, baud)
print("Connected to", port)

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp_ms', 'grip_strength'])

    print("Logging data...")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors="ignore").strip()
            print(line)

            if ',' in line:
                try:
                    timestamp, grip = line.split(',')
                    data.append({'timestamp_ms': int(timestamp), 'grip_strength': int(grip)})
                    sample_count += 1
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
