# Ex.No:04 FIT ARMA MODEL FOR TIME SERIES
# Date: 22.09.2025

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```python
###Name: Hycinth D
###Reg No: 212223240055
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
df = pd.read_csv("/mnt/data/Amazon.csv")

# Convert Date column to datetime and set as index
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Use Adjusted Close as the time series, enforce business-day frequency
X = df["Adj Close"].asfreq("B")

# Log transform (stabilize variance)
X_log = np.log(X)

# First difference to achieve stationarity
X_log_diff = X_log.diff().dropna()

# Plot stationary series
plt.figure(figsize=(12,6))
plt.plot(X_log_diff, label="Log Differenced Adj Close")
plt.title("Stationary Amazon Time Series")
plt.xlabel("Date")
plt.ylabel("Log Diff Price")
plt.legend()
plt.show()

# Check stationarity (ADF Test)
result = adfuller(X_log_diff)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Plot ACF and PACF
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plot_acf(X_log_diff, lags=40, ax=plt.gca())
plt.title("Stationary Data ACF")
plt.subplot(2,1,2)
plot_pacf(X_log_diff, lags=40, ax=plt.gca())
plt.title("Stationary Data PACF")
plt.tight_layout()
plt.show()

# Fit ARMA(1,1) Model
arma11_model = ARIMA(X_log_diff, order=(1, 0, 1)).fit()
print("ARMA(1,1) Summary:\n", arma11_model.summary())

# Simulate ARMA(1,1) process
phi1 = arma11_model.params.get('ar.L1', 0)
theta1 = arma11_model.params.get('ma.L1', 0)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])
arma11_sim = ArmaProcess(ar1, ma1).generate_sample(nsample=200)

plt.plot(arma11_sim)
plt.title("Simulated ARMA(1,1) Process")
plt.show()
plot_acf(arma11_sim)
plt.show()
plot_pacf(arma11_sim)
plt.show()

# Fit ARMA(2,2) Model
arma22_model = ARIMA(X_log_diff, order=(2, 0, 2)).fit()
print("ARMA(2,2) Summary:\n", arma22_model.summary())

# Simulate ARMA(2,2) process
phi1 = arma22_model.params.get('ar.L1', 0)
phi2 = arma22_model.params.get('ar.L2', 0)
theta1 = arma22_model.params.get('ma.L1', 0)
theta2 = arma22_model.params.get('ma.L2', 0)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])
arma22_sim = ArmaProcess(ar2, ma2).generate_sample(nsample=200)

plt.plot(arma22_sim)
plt.title("Simulated ARMA(2,2) Process")
plt.show()
plot_acf(arma22_sim)
plt.show()
plot_pacf(arma22_sim)
plt.show()


```

### OUTPUT:

### SIMULATED ARMA(1,1) PROCESS:

<img width="747" height="440" alt="image" src="https://github.com/user-attachments/assets/1e2994c3-cec7-453f-b62c-58831ae0fb83" />

<br>
<br>
<img width="541" height="434" alt="image" src="https://github.com/user-attachments/assets/a0f16b9b-a221-4e18-9acd-e19e1bf8c63d" />
<br>
<br>

### Partial Autocorrelation
<br>
<br>

<img width="556" height="430" alt="image" src="https://github.com/user-attachments/assets/09b0c4c0-cef6-4266-b880-fd59e3c45006" />

<br>
<br>

### Autocorrelation
<br>
<br>
<img width="557" height="442" alt="image" src="https://github.com/user-attachments/assets/a753c0fa-b27a-40cb-a0c5-fdbf9f799790" />
<br>
<br>

### SIMULATED ARMA(2,2) PROCESS:
<br>
<br>
<img width="710" height="507" alt="image" src="https://github.com/user-attachments/assets/01e70658-f8da-4f5a-8b12-b19bf9f3e5ac" />

<br>
<br>
<img width="537" height="434" alt="image" src="https://github.com/user-attachments/assets/f5db1b4b-9c88-43e8-a8a9-702c68803cfb" />

<br>
<br>

### Partial Autocorrelation
<br>
<br>
<img width="562" height="436" alt="image" src="https://github.com/user-attachments/assets/686c2736-70c0-4a06-912c-0c7a4f56ffd4" />

<br>
<br>


### Autocorrelation
<br>
<br>
<img width="560" height="443" alt="image" src="https://github.com/user-attachments/assets/6a07ab27-1c9b-4159-b31a-ac77284bbabe" />

<br>



### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
