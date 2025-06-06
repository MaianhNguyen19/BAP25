import threading
import time
import csv
import signal
import numpy as np
import dwfpy as dwf
import datetime
import serial

# === Parameters ===
Record_time = 20
SAMPLE_RATE = 1000  # Hz
BUFFER_SIZE = 10000   # Samples
READ_INTERVAL = 1*(BUFFER_SIZE/SAMPLE_RATE)  # Seconds (50 ms)
CSV_FILE = "sannegrip.csv"

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

        for ch in [0, 1]:
            scope[ch].setup(range=5)
            scope[1].setup(range=2)

        scope.sample_rate = SAMPLE_RATE
        scope.buffer_size = BUFFER_SIZE
        scope.scan_shift(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE, configure=True, start=True)

        # Initialize time tracking
        dt = 1 / SAMPLE_RATE
        sample_index = 0
        start_time = time.perf_counter()

        while running:
            time.sleep(READ_INTERVAL)
            scope.read_status(read_data=True)
            ch1 = np.array(scope[0].get_data())
            ch2 = np.array(scope[1].get_data())

            # Compute timestamps from sample index
            ts_start = start_time + sample_index * dt
            osc_queue.append((ts_start, ch1.tolist(), ch2.tolist()))

            sample_index += len(ch1)


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
# Define stop_all so it works for both signal and timer
def stop_all(sig=None, frame=None):
    global running
    print("Stopping recording...")
    running = False

# Register Ctrl+C handler
signal.signal(signal.SIGINT, stop_all)

# Start auto-stop timer
stop_timer = threading.Timer(Record_time, stop_all)
stop_timer.start()

signal.signal(signal.SIGINT, stop_all)

# === Data Saver (Main Thread) ===
def gather_and_save():
    all_rows = []
    osc_sample_rate = SAMPLE_RATE  # same as used in record_oscilloscope

    print("Logging to CSV... Press Ctrl+C to stop.")

    empty_loops = 0
    max_empty_loops = 40  # ~2 seconds idle

    while True:
        wrote_data = False

        # === Handle oscilloscope batches ===
        while osc_queue:
            ts, ch1, ch2 = osc_queue.pop(0)
            dt = 1 / SAMPLE_RATE
            for i in range(len(ch1)):
                t_sample = ts + i * dt
                all_rows.append([t_sample, ch1[i], ch2[i], None, None])
            wrote_data = True


        # === Handle ESP32 real-time samples ===
        while esp_queue:
            host_ts, val = esp_queue.pop(0)
            esp_time, sensor_val = val
            # You can choose to use `host_ts` or `esp_time`
            all_rows.append([host_ts, None, None, esp_time, sensor_val])

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
        writer.writerow(["timestamp", "osc_ch1", "osc_ch2", "esp_timestamp", "esp_value"])
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


