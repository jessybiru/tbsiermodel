import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seird_model(y, t, beta, sigma, gamma, mu, mu_tb, mu_birth):
    S, E, I, R, D = y
    
    dSdt = mu_birth - mu * S - beta * S * I
    dEdt = beta * S * I - (sigma + mu) * E
    dIdt = sigma * E - (gamma + mu + mu_tb) * I
    dRdt = gamma * I - mu * R
    dDdt = mu_tb * I
    
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# Initial conditions
S0 = 1000000  # Initial number of susceptible individuals
E0 = 100       # Initial number of exposed individuals
I0 = 50        # Initial number of infectious individuals
R0 = 0         # Initial number of recovered individuals
D0 = 0         # Initial number of deaths


# Parameters
beta = 7    # Contact rate (rate of transmission)
sigma = 0.143   # Rate of progression from exposed to infectious
gamma = 0.167  # Recovery rate 
mu = 0.0056    # Natural death rate
mu_tb = 0.14 # Death rate due to TB
mu_birth = 0.035  # Birth rate


# Time vector
# Time vector
months = 12
t = np.linspace(0, months, num=30*months)  # Time span of 1 year with monthly time steps

# Initial conditions vector
y0 = [S0, E0, I0, R0, D0]

# Integrate the SEIRD equations over the time grid using odeint
result = odeint(seird_model, y0, t, args=(beta, sigma, gamma, mu, mu_tb, mu_birth))
S, E, I, R, D = result.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deaths')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIRD Model for TB Transmission in Uganda')
plt.legend()
plt.grid(True)
plt.show()
