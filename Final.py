import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
def oscillatory_function(t, A, omega, phase, B):
    return A * np.sin(omega * t + phase) + B
# Add in the next part, (The data part, and the safe_convert_time)
data = pd.read_csv('ASTR19_F23_group_project_data.txt', delim_whitespace=True, header=None, names=['Day', 'Time', 'TideHeight'],comment='#')
def safe_convert_time(time_str):
    try:
        return pd.to_datetime(time_str, format='%H:%M').time()
    except ValueError:
        return None  # or some default value
data['Time'] = data['Time'].apply(safe_convert_time)
data = data.dropna(subset=['Time'])

data['Time'] = data['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60) / 86400.0)
data['TotalTime'] = data['Day'] + data['Time']


sigma = np.full(len(data['TideHeight']), 0.25)
params, params_covariance = curve_fit(oscillatory_function, data['TotalTime'], data['TideHeight'], sigma=sigma)


plt.figure(figsize=(10, 5))
plt.scatter(data['TotalTime'], data['TideHeight'], label='Data', color='blue')
plt.plot(data['TotalTime'], oscillatory_function(data['TotalTime'], *params), label='Fitted model', color='red')
plt.xlabel('Time (days)')
plt.ylabel('Tide Height (ft)')
plt.title('Tide Height Data and Fitted Oscillatory Model')
plt.legend()
plt.grid(True)
plt.savefig('tide_model_fit.pdf')
plt.show()