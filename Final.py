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

residuals = data['TideHeight'] - oscillatory_function(data['TotalTime'], *params)
plt.figure(figsize=(10, 5))
plt.scatter(data['TotalTime'], residuals, color='blue', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
plt.xlabel('Time (days)')
plt.ylabel('Residuals (ft)')
plt.title('Residuals of Tide Height Data')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('residuals_tide_height.pdf')

plt.figure(figsize=(10, 5))
bin_width = 0.1  # Choosing a reasonable bin width
bins = np.arange(min(residuals), max(residuals) + bin_width, bin_width)
plt.hist(residuals, bins=bins, color='skyblue', edgecolor='black', label='Residuals')
plt.xlabel('Residuals (ft)')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.legend()
plt.show()
plt.savefig('residuals_histogram.pdf')

std_dev = np.std(residuals)
print(f"Standard Deviation of Residuals: {std_dev} ft")
experimental_error = 0.25  # ft, as given
print(f"Assumed Experimental Error: {experimental_error} ft")
intrinsic_scatter = np.sqrt(max(std_dev**2 - experimental_error**2, 0))
print(f"Intrinsic Scatter: {intrinsic_scatter} ft")

jan14_times = data[data['Day'] == 14]['TotalTime']
jan14_predicted_tides = oscillatory_function(jan14_times.values, *params)  # Ensure it's an array
first_high_tide_index = jan14_predicted_tides.argmax()  # Index in the filtered array
first_high_tide_time = jan14_times.iloc[first_high_tide_index]
tsunami_residual = 2 + oscillatory_function(first_high_tide_time, *params) - oscillatory_function(first_high_tide_time, *params)
tsunami_deviation_std = tsunami_residual / std_dev
print(f"Tsunami Deviation: {tsunami_deviation_std:.2f} standard deviations")
residuals_with_tsunami = np.append(residuals, tsunami_residual)
plt.figure(figsize=(10, 5))
plt.hist(residuals_with_tsunami, bins=bins, color='skyblue', edgecolor='black', label='Residuals with Tsunami')
plt.axvline(tsunami_residual, color='red', linestyle='--', label='Tsunami Residual')
plt.xlabel('Residuals (ft)')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals with Tsunami Outlier')
plt.legend()
plt.show()
plt.savefig('residuals_histogram_with_tsunami.pdf')