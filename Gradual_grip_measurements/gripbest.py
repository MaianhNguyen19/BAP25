import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter, hilbert
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# === Parameters ===
fs = 1000                # sampling rate (Hz)
drop_ms = 200            # drop first 200 ms of startup
V_SUPPLY = 3.3           # CH1 supply voltage
R_FIXED = 2.7e3          # CH1 fixed resistor (Ω)
lag_samples = 297        # CH2 envelope leads CH1 by ~0.297 s

# CH1 calibration (power‐law)
weights_calib     = np.array([128, 256, 259, 518, 2*128+518, 1000], float)
resistances_calib = np.array([20e3, 10e3, 9.1e3, 5e3, 2.7e3, 2.1e3], float)
b, log_a          = np.polyfit(np.log(weights_calib), np.log(resistances_calib), 1)
a                 = np.exp(log_a)
def resistance_from_voltage(V): return R_FIXED * (V_SUPPLY / V - 1)
def weight_from_voltage(V):
    Rs = resistance_from_voltage(V)
    return (Rs / a)**(1.0 / b)

# Filters
bp_taps = firwin(200, [45, 55], pass_zero=False, fs=fs, window='blackman')
lp_taps = firwin(101, 5.0, fs=fs)

def process_file(path):
    df = pd.read_csv(path).dropna(subset=['osc_ch1','osc_ch2'])
    # timestamp zero‐reference
    df['t']     = df['timestamp'].astype(float) - df['timestamp'].iloc[0]
    # voltages
    df['ch1_v'] = df['osc_ch1'] * 10.0
    df['ch2_v'] = df['osc_ch2'] * 30.0
    # drop invalid CH1 and startup
    drop_n = int(fs * drop_ms / 1000)
    df = df[(df.ch1_v>0)&(df.ch1_v<V_SUPPLY)].iloc[drop_n:].reset_index(drop=True)

    # CH1 → true weight
    ch1 = lfilter(lp_taps, 1.0, df.ch1_v.values)
    ch1 = lfilter(lp_taps, 1.0, ch1[::-1])[::-1]
    weight_true = weight_from_voltage(ch1)

    # CH2 envelope
    ch2 = lfilter(bp_taps, 1.0, df.ch2_v.values)
    ch2 = lfilter(bp_taps, 1.0, ch2[::-1])[::-1]
    env = np.abs(hilbert(ch2))
    env = lfilter(lp_taps, 1.0, env)
    env = lfilter(lp_taps, 1.0, env[::-1])[::-1]

    # align by lag
    env_a = np.roll(env, -lag_samples)[:-lag_samples]
    w_a   = weight_true[lag_samples:lag_samples+len(env_a)]

    # metrics
    r, _       = pearsonr(env_a, w_a)
    r2         = r**2
    rmse       = np.sqrt(mean_squared_error(w_a, env_a))  # or your model-prediction
    norm_rmse  = rmse / (w_a.max() - w_a.min()) * 100

    return {
        'file':         path,
        'Pearson r':    round(r, 3),
        'R²':           round(r2, 3),
        'Norm. RMSE (%)': round(norm_rmse, 1)
    }

if __name__ == '__main__':
    files = ['jeanrgrip.csv', 'jeanlgrip.csv','stephlgrip.csv','stephrgrip.csv','daivdlgrip.csv','daivdrgrip.csv']
    results = [process_file(f) for f in files]
    df_res = pd.DataFrame(results).set_index('file')
    print(df_res.to_markdown())