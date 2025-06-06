import threading
import time
import csv
import signal
import numpy as np
import dwfpy as dwf
import datetime
import serial

# === Parameters ===
SAMPLE_RATE = 1000  # Hz
BUFFER_SIZE = 1000   # Samples
READ_INTERVAL = 0.95*(BUFFER_SIZE/SAMPLE_RATE)  # Seconds (50 ms)
CSV_FILE = "experiment_log.csv"

# === Thread control flag ===
running = True

# === Shared queue to pass data from thread to logger ===
osc_queue = []
esp_queue = []

# === Oscilloscope Thread Function ===
def record_oscilloscope():
    with dwf.Device() as device:
        print(f"Connected to: {device.name} ({device.serial_number})")
        scope = device.analog_input
        scope[0].setup(range=50)
        scope.sample_rate = SAMPLE_RATE
        scope.buffer_size = BUFFER_SIZE
        scope.scan_shift(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE, configure=True, start=True)

        while running:
            time.sleep(READ_INTERVAL)
            scope.read_status(read_data=True)
            samples = np.array(scope[0].get_data())
            ts = time.perf_counter()-(len(samples)/SAMPLE_RATE)
            osc_queue.append((ts, samples.tolist()))


def record_esp32_serial(port="COM5", baud=921600):
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
        print(f"Connected to ESP32 on {port} @ {baud} baud.")
    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port: {e}")
        return

    try:
        while running:
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    try:
                        parts = [int(x.strip()) for x in line.split(',')]
                        if len(parts) == 2:
                            esp_time = parts[0] / 1_000_000  # µs to sec
                            val = parts[1]
                            host_time = time.perf_counter()
                            esp_queue.append((host_time, [esp_time, val]))
                        else:
                            print(f"[ESP PARSE WARNING] Invalid line: {line}")
                    except Exception as e:
                        print(f"[ESP PARSE ERROR] Line: {line} → {e}")
            except Exception as read_err:
                print(f"[ESP32 Read Error] {read_err}")
                break
    finally:
        ser.close()
        print("ESP32 serial closed.")
    

# === Signal handler for Ctrl+C ===
def stop_all(sig, frame):
    global running
    print("Stopping recording...")
    running = False

signal.signal(signal.SIGINT, stop_all)

# === Data Saver (Main Thread) ===
def gather_and_save():
    all_rows = []
    osc_sample_rate = SAMPLE_RATE  # same as used in record_oscilloscope

    print("Logging to CSV... Press Ctrl+C to stop.")

    empty_loops = 0
    max_empty_loops = 400  # ~2 seconds idle

    while True:
        wrote_data = False

        # === Handle oscilloscope batches ===
        while osc_queue:
            ts, samples = osc_queue.pop(0)
            dt = 1 / osc_sample_rate
            for i, val in enumerate(samples):
                ts_sample = ts + i * dt
                all_rows.append([ts_sample, val, None])
            wrote_data = True

        # === Handle ESP32 real-time samples ===
        while esp_queue:
            host_ts, val = esp_queue.pop(0)
            esp_time, sensor_val = val
            # You can choose to use `host_ts` or `esp_time`
            all_rows.append([host_ts, None, esp_time, sensor_val])

            wrote_data = True

        if not wrote_data:
            empty_loops += 1
            if not running and empty_loops > max_empty_loops:
                print("Queues idle and stopped. Exiting logger.")
                break
        else:
            empty_loops = 0

        time.sleep(0.005)

    # === Sort by timestamp ===
    all_rows.sort(key=lambda row: row[0])

    # === Write to CSV ===
    with open(CSV_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "osc_value", "esp_timestamp", "esp_value"])
        writer.writerows(all_rows)

    print("Recording complete.")

osc_thread = threading.Thread(target=record_oscilloscope)
esp_thread = threading.Thread(target=record_esp32_serial)

osc_thread.start()
esp_thread.start()

gather_and_save()

osc_thread.join()
esp_thread.join()
print("All threads joined. Program ended.")


