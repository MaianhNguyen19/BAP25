import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('grip_data_1.csv')
df = df.head(100)

# Plot grip vs time
plt.figure(figsize=(10, 5))
plt.plot(df['relative_time_us'], df['grip_voltage'], marker='o', linestyle='-')
plt.xlabel('Time (microseconds)')
plt.ylabel('Grip Strength')
plt.title('Grip Strength Over First X Samples')
plt.grid(True)
plt.tight_layout()
plt.show()