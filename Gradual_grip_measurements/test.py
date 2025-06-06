import threading
import time
import csv
import signal
import numpy as np
import dwfpy as dwf
import datetime
import serial

# === Parameters ===
SAMPLE_RATE = 5000  # Hz
BUFFER_SIZE = 512   # Samples
READ_INTERVAL = 0.05  # Seconds (50 ms)
CSV_FILE = "experiment_log.csv"

# === Thread control flag ===
running = True

# === Shared queue to pass data from thread to logger ===
osc_queue = []

# === Oscilloscope Thread Function ===
def record_oscilloscope():
    with dwf.Device() as device:
        print(f"Connected to: {device.name} ({device.serial_number})")
        scope = device.analog_input
        scope[0].setup(range=5.0)
        scope.sample_rate = SAMPLE_RATE
        scope.buffer_size = BUFFER_SIZE
        scope.scan_shift(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE, configure=True, start=True)

        while running:
            time.sleep(READ_INTERVAL)
            scope.read_status(read_data=True)
            samples = np.array(scope[0].get_data())
            ts = time.perf_counter()
            osc_queue.append((ts, samples.tolist()))


def record_esp32_serial(port="COM5", baud=921600):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"Connected to ESP32 on {port} @ {baud} baud.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return

    while running:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                ts = time.perf_counter()
                esp_queue.append((ts, line))
        except Exception as e:
            print(f"Read error: {e}")
            break

    ser.close()
        

# === Signal handler for Ctrl+C ===
def stop_all(sig, frame):
    global running
    print("Stopping recording...")
    running = False

signal.signal(signal.SIGINT, stop_all)

# === Data Saver (Main Thread) ===
def gather_and_save():
    with open(CSV_FILE, "w", newline='') as f:
        # Write metadata as commented header lines
        f.write(f"# Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"# Buffer size: {BUFFER_SIZE} samples\n")
        f.write(f"# Read interval: {READ_INTERVAL} seconds\n")
        f.write(f"# Recording start: {datetime.datetime.now().isoformat()}\n")

        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])  # CSV header

        print("Logging to CSV... Press Ctrl+C to stop.")
        while running or len(osc_queue) > 0:
            if osc_queue:
                ts, samples = osc_queue.pop(0)
                for val in samples:
                    writer.writerow([ts, val])
            else:
                time.sleep(0.005)

        print(f"Recording complete. Data saved to: {CSV_FILE}")

# === Start threads ===
osc_thread = threading.Thread(target=record_oscilloscope)
osc_thread.start()

# Run main saver loop in main thread
gather_and_save()

osc_thread.join()
