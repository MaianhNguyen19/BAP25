import pandas as pd

# Load the CSV
df = pd.read_csv("calibration_data.csv")

# Group by weight and take the average sensor value
avg_data = df.groupby('Total Weight (g)')['Sensor Value (V)'].mean().reset_index()

print(avg_data)

from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = avg_data['Sensor Value (V)'].values.reshape(-1, 1)  # voltage
y = avg_data['Total Weight (g)'].values  # weight in grams

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Show model
print(f"Weight = {model.coef_[0]:.2f} × Voltage + {model.intercept_:.2f}")

new_voltage = 2.45  # example sensor reading
predicted_weight = model.predict([[new_voltage]])[0]
print(f"Predicted weight: {predicted_weight:.2f} g")

import matplotlib.pyplot as plt

plt.scatter(X, y, label='Measured')
plt.plot(X, model.predict(X), color='red', label='Fit')
plt.xlabel("Voltage (V)")
plt.ylabel("Weight (g)")
plt.title("Calibration Curve: Voltage to Weight")
plt.legend()
plt.grid(True)
plt.show()
