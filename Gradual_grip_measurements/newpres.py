import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# === File list: add your CSV filenames here ===
csv_files = ["graddavid.csv", "gradsanne.csv", "gradmath.csv"]  # replace with actual names

# === 1) CH1 weight calibration (power-law) ===
weights_calib     = np.array([128, 256, 259, 518, (2*128 + 518), 1000])  # grams
resistances_calib = np.array([20e3, 10e3,  9.1e3, 5e3,   2.7e3,       2.1e3])  # Ω

# Fit R_sensor = a · w^b
b, log_a = np.polyfit(np.log(weights_calib), np.log(resistances_calib), 1)
a = np.exp(log_a)
print(f"Calibration: R_sensor(w) = {a:.3e}·w^{b:.3f}   (Ω, w in g)")

R_FIXED = 2.7e3   # Ω
V_SUPPLY = 3.3    # V

def resistance_from_voltage_ch1(V_ch1):
    return R_FIXED * (V_SUPPLY / V_ch1 - 1)

def weight_from_voltage_ch1(V_ch1):
    R_pred = resistance_from_voltage_ch1(V_ch1)
    return (R_pred / a) ** (1.0 / b)

# === Pre-allocate lists for combined plotting ===
all_data = []

for csv_path in csv_files:
    # === 2) Load CSV and clean ===
    df = pd.read_csv(csv_path)
    osc = df.dropna(subset=["osc_ch1", "osc_ch2"]).copy()
    osc["timestamp"] = osc["timestamp"].astype(float)
    osc["timestamp"] -= osc["timestamp"].iloc[0]
    osc["ch1_volts"] = osc["osc_ch1"] * 10.0
    osc["ch2_volts"] = osc["osc_ch2"] * 30.0
    mask_valid = (osc["ch1_volts"] > 0) & (osc["ch1_volts"] < V_SUPPLY)
    osc = osc[mask_valid].reset_index(drop=True)

    # === 3) Smooth CH1 voltage ===
    fs = 1000
    taps_ch1 = firwin(101, 5.0, fs=fs, window="hamming")
    sm = lfilter(taps_ch1, 1.0, osc["ch1_volts"].values)
    sm = lfilter(taps_ch1, 1.0, sm[::-1])[::-1]
    osc["ch1_smoothed"] = sm
    osc["weight_g"] = osc["ch1_smoothed"].apply(weight_from_voltage_ch1)

    # === 5) CH2 bandpass and RMS ===
    def blackman_bandpass_filter(data, lowcut, highcut, fs, numtaps):
        taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs, window="blackman")
        return lfilter(taps, 1.0, data)
    def moving_rms(sig, ws):
        return np.array([np.sqrt(np.mean(sig[i:i+ws]**2)) 
                         for i in range(len(sig)-ws+1)])
    filtered = blackman_bandpass_filter(osc["ch2_volts"].values, 40, 60, fs, 30)
    win_fast = int(0.05 * fs)
    rms_fast = moving_rms(filtered, win_fast)
    t_rms = osc["timestamp"].iloc[:len(rms_fast)].values
    # Trim weight to same length
    w_trim = osc["weight_g"].iloc[:len(rms_fast)].values

    # === 6) EMA smoothing ===
    tau = 0.2
    alpha = 1 - np.exp(-1/(tau*fs))
    rms_slow = np.zeros_like(rms_fast)
    rms_slow[0] = rms_fast[0]
    for i in range(1, len(rms_fast)):
        rms_slow[i] = alpha * rms_fast[i] + (1-alpha) * rms_slow[i-1]

    # store processed data
    all_data.append({
        'label': csv_path,
        'time': osc['timestamp'],
        'ch1_raw': osc['ch1_volts'],
        'ch1_smooth': osc['ch1_smoothed'],
        'weight': osc['weight_g'],
        't_rms': t_rms,
        'rms_fast': rms_fast,
        'rms_slow': rms_slow,
        'weight_trim': w_trim
    })

# === 8) Plot all datasets together ===
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

for data in all_data:
    axes[0].plot(data['time'], data['ch1_raw'], alpha=0.4, label=f"{data['label']} raw")
    axes[0].plot(data['time'], data['ch1_smooth'], label=f"{data['label']} smooth")
axes[0].set(title="CH1: Raw vs Smoothed", xlabel="Time (s)", ylabel="Voltage (V)")
axes[0].legend(); axes[0].grid(True)

for data in all_data:
    axes[1].plot(data['time'], data['weight'], label=data['label'])
axes[1].set(title="Weight vs Time", xlabel="Time (s)", ylabel="Weight (g)")
axes[1].legend(); axes[1].grid(True)

for data in all_data:
    axes[2].plot(data['t_rms'], data['rms_fast'], label=f"{data['label']} fast")
    axes[2].plot(data['t_rms'], data['rms_slow'], label=f"{data['label']} slow")
axes[2].set(title="CH2 RMS Fast vs Slow", xlabel="Time (s)", ylabel="RMS (V)")
axes[2].legend(); axes[2].grid(True)

# Scatter and fit only for first dataset as example
sample = all_data[0]
axes[3].plot( osc["timestamp"], osc["ch2_volts"], label=f"{data['label']} fast")
axes[3].plot( osc["timestamp"], osc["ch1_volts"], label=f"{data['label']} slow")
axes[3].set(title="RMS_slow vs Weight (log–log)", xlabel="RMS_slow (V)", ylabel="Weight (g)")
axes[3].grid(True, which='both', ls='--', alpha=0.5)


plt.tight_layout()
plt.show()
Multivariate()