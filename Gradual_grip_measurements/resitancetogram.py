import numpy as np
import matplotlib.pyplot as plt

# 1) Original data: weights in grams and corresponding sensor resistances in ohms
weights = np.array([128, 256, 259, 518, (2*128 + 518), 1000])  # grams
resistances = np.array([20e3, 10e3, 9.1e3, 5e3, 2.7e3, 2.1e3])  # ohms

# 2) Fit a power‐law model: R_sensor ≈ a * (weight)^b
#    Taking logs: log(R) = log(a) + b * log(weight)
log_w = np.log(weights)
log_R = np.log(resistances)
b, log_a = np.polyfit(log_w, log_R, 1)
a = np.exp(log_a)
print(f"Fitted model: R_sensor(w) = {a:.3e} * w^{b:.3f}")

# 3) Function to predict sensor resistance from a given weight (in grams)
def predict_resistance_from_weight(w):
    """
    Given weight w in grams, returns predicted sensor resistance (Ω)
    based on the power‐law fit.
    """
    return a * (w ** b)

# 4) Voltage‐divider conversion parameters
R_FIXED = 10e3   # 10 kΩ fixed resistor
V_SUPPLY = 3.3   # 3.3 V source

def voltage_from_resistance(R_sensor):
    """
    Given the sensor resistance R_sensor (Ω) in series with R_FIXED (10 kΩ),
    returns the voltage at the midpoint of the divider.
    """
    return V_SUPPLY * (R_FIXED / (R_sensor + R_FIXED))

# 5) Prompt the user for an input weight (in grams), compute predicted R and V
try:
    user_input = input("Enter a weight value in grams (e.g., 500): ").strip()
    w_input = float(user_input)
    if w_input <= 0:
        raise ValueError("Weight must be positive.")
except Exception as e:
    print(f"Invalid input: {e}")
    exit(1)

R_pred = predict_resistance_from_weight(w_input)
V_pred = voltage_from_resistance(R_pred)

print(f"\nFor weight = {w_input:.1f} g:")
print(f"  Predicted Sensor Resistance ≈ {R_pred:.1f} Ω")
print(f"  Predicted Divider Voltage   ≈ {V_pred:.3f} V")

# 6) (Optional) Plot the fitted curve and highlight the user's input
w_fit = np.logspace(np.log10(weights.min()), np.log10(weights.max()), 200)
R_fit = predict_resistance_from_weight(w_fit)
V_fit = voltage_from_resistance(R_fit)

plt.figure(figsize=(5, 4))
plt.loglog(weights, resistances, 'o', label="Original data")
plt.loglog(w_fit, R_fit, '-', label=f"Fit: R = {a:.2e}·w^{b:.2f}")
plt.loglog(w_input, R_pred, 's', markersize=8, label="Your input → R_pred", color='C1')
plt.xlabel("Weight (g) [log scale]")
plt.ylabel("Resistance (Ω) [log scale]")
plt.title("Sensor Resistance vs. Weight")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.loglog(w_fit, V_fit, '-', label="Voltage vs. Weight")
plt.loglog(w_input, V_pred, 's', markersize=8, label="Your input → V_pred", color='C1')
plt.xlabel("Weight (g) [log scale]")
plt.ylabel("Voltage (V) [log scale]")
plt.title("Divider Voltage vs. Weight")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
