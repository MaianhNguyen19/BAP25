import serial
import time

# -------- Setup Serial Port --------
port = 'com5'  # Change to your Arduino port
baudrate = 9600        # Match your Arduino
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(2)  # Allow Arduino to reset

# -------- Prediction Function --------
def predict_weight(voltage):
    return (
        272.5
        + 0.153 * voltage
        - 0.000132 * voltage**2
        + 0.0000000986 * voltage**3
    )


print("🔄 Reading sensor data and predicting weight...\n(Press Ctrl+C to stop)\n")

# -------- Main Loop --------
try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                voltage = float(line)
                weight = predict_weight(voltage)
                print(f"Voltage: {voltage:.3f} V\tEstimated Weight: {weight:.2f} g")
            except ValueError:
                print(f"⚠️  Skipped invalid input: '{line}'")
except KeyboardInterrupt:
    print("\n🔚 Stopped by user.")
finally:
    ser.close()
