#!/usr/bin/env python3
import time
import serial
import numpy as np

# ——— CONFIG ———
PORT       = "COM5"
BAUD       = 921600
DURATION   = 10        # seconds to capture
CSV_OUT    = "esp32_test1.csv"
# ——————————

def main():
    times = []
    vals  = []

    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Listening on {PORT} @ {BAUD}, for {DURATION}s…")

    t0 = time.time()
    first_us = None
    while time.time() - t0 < DURATION:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        try:
            t_us = int(parts[0])
            v    = int(parts[1])
        except ValueError:
            continue

        if first_us is None:
            first_us = t_us
        t_rel = (t_us - first_us)/1e6

        times.append(t_rel)
        vals.append(v)

    ser.close()
    print("Done capturing.")
    n = len(times)
    print(f"Samples received: {n}")

    # Compute Δt stats
    dts = np.diff(times)
    print(f"Mean Δt = {np.mean(dts)*1e3:.3f} ms,  Std Δt = {np.std(dts)*1e3:.3f} ms")
    print(f"Min Δt = {np.min(dts)*1e3:.3f} ms,  Max Δt = {np.max(dts)*1e3:.3f} ms")

    # Write a tiny CSV
    import csv
    with open(CSV_OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Time_s","ADC6"])
        w.writerows(zip(times, vals))
    print(f"Wrote data to {CSV_OUT}")

if __name__ == "__main__":
    main()
