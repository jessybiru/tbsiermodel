
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seird_model(y, t, beta, sigma, gamma_no_dots, gamma_dots, mu, mu_tb, mu_birth, dots_coverage, treatment_start_time):
    S, E, I, R, D = y

    dSdt = mu_birth - mu * S - beta * S * I
    dEdt = beta * S * I - (sigma + mu) * E
    dIdt = sigma * E - (get_gamma(t, gamma_no_dots, gamma_dots, treatment_start_time) + mu + mu_tb) * I
    dRdt = (1 - dots_coverage) * get_gamma(t, gamma_no_dots, gamma_dots, treatment_start_time) * I - mu * R
    dDdt = dots_coverage * mu_tb * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def get_gamma(t, gamma_no_dots, gamma_dots, treatment_start_time):
    # Time-dependent recovery rate function
    return gamma_no_dots if t < treatment_start_time else gamma_dots

#Initial conditions
S0 = 1080 # Initial number of susceptible individuals
E0 = 20   # Initial number of exposed individuals
I0 = 4   # Initial number of infectious individuals
R0 = 0        # Initial number of recovered individuals
D0 = 0         # Initial number of deaths

# Parameters
beta = 0.58   # Contact rate (rate of transmission)
sigma = 0.143   # Rate of progression from exposed to infectious
gamma_no_dots = 0.167  # Recovery rate without DOTS
gamma_dots = 0.74       # Recovery rate with DOTS
mu = 0.0056    # Natural death rate
mu_tb = 0.14 # Death rate due to TB
mu_birth = 0.035  # Birth rate

# Time vector
months =12
t = np.linspace(0, months, num=30*months)  # Time span of 1 year with monthly time steps

#dots coverage
dots_coverage = 0.6  # DOTS coverage (percentage of infectious individuals receiving DOTS treatment)
treatment_start_time = 4  # Time in months when DOTS treatment starts

# Initial conditions vector
y0 = [S0, E0, I0, R0, D0]

#Integrate the SEIRD equations over the time grid using  odeint
result = odeint(seird_model, y0, t, args=(beta, sigma, gamma_no_dots, gamma_dots, mu, mu_tb, mu_birth, dots_coverage, treatment_start_time))
S, E, I, R, D = result.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deaths')
plt.xlabel('Time (months)')
plt.ylabel('Population')
plt.title('SEIRD Model for TB Transmission with DOTS Intervention in Uganda')
plt.legend()
plt.grid(True)
plt.show()

