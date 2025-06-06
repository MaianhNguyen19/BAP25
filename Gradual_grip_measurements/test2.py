import threading
import time
import numpy as np
import dwfpy as dwf

# Shared state
running = True

def record_oscilloscope():
    with dwf.Device() as device:
        print(f"Connected to: {device.name} ({device.serial_number})")
        scope = device.analog_input
        scope[0].setup(range=5.0)
        scope.sample_rate = 10000
        scope.buffer_size = 1024
        scope.configure(start=True)

        while running:
            time.sleep(0.05)
            scope.read_status(read_data=True)
            samples = np.array(scope[0].get_data())
            #print(f"[{time.perf_counter():.3f}] Got {len(samples)} samples — mean: {np.mean(samples):.3f}")

        scope.stop()

# Handle Ctrl+C to stop gracefully
import signal
def stop_all(sig, frame):
    global running
    print("Stopping oscilloscope thread...")
    running = False

signal.signal(signal.SIGINT, stop_all)

# Start oscilloscope thread
osc_thread = threading.Thread(target=record_oscilloscope)
osc_thread.start()

# Main thread waits
osc_thread.join()
print("Done.")
