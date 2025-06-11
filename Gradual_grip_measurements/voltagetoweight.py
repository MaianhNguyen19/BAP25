import numpy as np
import matplotlib.pyplot as plt

# 1) Calibration data: weights in grams and corresponding sensor resistances in ohms
weights = np.array([128, 256, 259, 518, (2*128 + 518), 1000])  # grams
resistances = np.array([20e3, 10e3, 9.1e3, 5e3, 2.7e3, 2.1e3])  # ohms

# 2) Fit a power‐law model R = a * w^b via log–log linear regression
log_w = np.log(weights)
log_R = np.log(resistances)
b, log_a = np.polyfit(log_w, log_R, 1)
a = np.exp(log_a)
print(f"Fitted model: R_sensor(w) = {a:.3e} * w^{b:.3f}")

# 3) Forward functions
def predict_resistance_from_weight(w):
    return a * w**b

R_FIXED = 2.7e3   # 10 kΩ fixed resistor
V_SUPPLY = 3.3   # 3.3 V

def voltage_from_resistance(R_sensor):
    return V_SUPPLY * (R_FIXED / (R_FIXED + R_sensor))

# 4) Inverse functions
def resistance_from_voltage(V):
    # Invert the divider: V = V_SUPPLY * R_FIXED/(R_FIXED + R)
    return R_FIXED * (V_SUPPLY / V - 1)

def weight_from_resistance(R_sensor):
    # Invert the power-law: R = a * w^b  ⇒  w = (R/a)^(1/b)
    return (R_sensor / a)**(1/b)

def weight_from_voltage(V):
    R_est = resistance_from_voltage(V)
    return weight_from_resistance(R_est)

# 5) Example usage (replace this with your measured voltage)
V_measured = 1.2  # volts, for example
R_est = resistance_from_voltage(V_measured)
w_est = weight_from_voltage(V_measured)
print(f"\nFor measured voltage = {V_measured:.3f} V:")
print(f"  Estimated resistance = {R_est:.1f} Ω")
print(f"  Estimated weight     = {w_est:.1f} g")

# 6) Plot calibration and inverse mapping
w_fit = np.logspace(np.log10(weights.min()), np.log10(weights.max()), 200)
R_fit = predict_resistance_from_weight(w_fit)
V_fit = voltage_from_resistance(R_fit)

plt.figure(figsize=(5,4))
plt.loglog(weights, resistances, 'o', label='Cal. data')
plt.loglog(w_fit, R_fit, '-', label='Fit: R vs. w')
plt.loglog(w_est, R_est, 's', label='Input→R_est')
plt.xlabel('Weight (g)')
plt.ylabel('Resistance (Ω)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()

plt.figure(figsize=(5,4))
plt.semilogx(w_fit, V_fit, '-', label='V vs. w')
plt.semilogx(w_est, V_measured, 's', label='Input→w_est')
plt.xlabel('Weight (g)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Given your a, b, R_FIXED, V_SUPPLY from the fit
def weight_from_voltage(V):
    R_est = R_FIXED * (V_SUPPLY/V - 1)
    return (R_est / a)**(1/b)

V_vals = np.linspace(0.5, 2.5, 100)  # realistic voltage range
w_vals = weight_from_voltage(V_vals)

plt.figure()
plt.plot(V_vals, w_vals, '-')
plt.scatter([V_measured], [w_est], color='C1', label='Measured point')
plt.xlabel('Voltage (V)')
plt.ylabel('Estimated Weight (g)')
plt.title('Voltage → Weight Calibration')
plt.grid(True)
plt.legend()
plt.show()