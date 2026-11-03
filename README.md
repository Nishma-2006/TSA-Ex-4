
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES




### AIM:
To implement ARMA model in python.

Dataset used:
ice_cream

Software Required:
Google colab

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

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('/content/ice_cream.csv')

# Declare required variables and visualise data
N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

X = data['IPN31152N'].dropna()

plt.plot(X)
plt.title('Original Ice Cream Data')
plt.show()

# Plot ACF and PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=30, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=30, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# Fitting ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1, lags=30)
plt.title('ARMA(1,1) ACF')
plt.show()

plot_pacf(ARMA_1, lags=30)
plt.title('ARMA(1,1) PACF')
plt.show()

# Fitting ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2, lags=30)
plt.title('ARMA(2,2) ACF')
plt.show()

plot_pacf(ARMA_2, lags=30)
plt.title('ARMA(2,2) PACF')
plt.show()
```







## OUTPUT:

<img width="1648" height="934" alt="image" src="https://github.com/user-attachments/assets/c982a802-4ef1-4558-b3b3-d55a8560abb0" />
<img width="1773" height="345" alt="image" src="https://github.com/user-attachments/assets/745a2474-79ae-4b86-9c0e-a87ef2833613" />


<img width="1489" height="601" alt="image" src="https://github.com/user-attachments/assets/e52446c6-66c1-4d2c-9133-02da28103825" />


<img width="1551" height="601" alt="image" src="https://github.com/user-attachments/assets/ad448af8-1724-4499-b58e-ff907da8bf50" />

<img width="1417" height="586" alt="image" src="https://github.com/user-attachments/assets/f22ec7ac-e0a8-48f4-95b4-babad2b3a941" />

<img width="1614" height="1198" alt="image" src="https://github.com/user-attachments/assets/8e568f86-289b-4f93-a5d6-43bc58130454" />

<img width="1620" height="609" alt="image" src="https://github.com/user-attachments/assets/2bbb461d-0da4-42e2-8ad7-e139e195df01" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.




