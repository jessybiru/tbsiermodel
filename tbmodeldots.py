
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seird_model(y, t, beta, sigma, gamma_no_dots, mu, mu_tb, mu_birth, dots_coverage):
    S, E, I, R, D = y
    
    dSdt = mu_birth - mu * S - beta * S * I
    dEdt = beta * S * I - (sigma + mu) * E
    dIdt = sigma * E - (gamma_no_dots + mu + mu_tb) * I
    dRdt = (1 - dots_coverage) * gamma_no_dots * I - mu * R
    dDdt = dots_coverage * mu_tb * I
    
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

##sensitivity test

import numpy as np
from scipy.integrate import odeint

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

def seird_model(y, t, beta, sigma, gamma_no_dots, gamma_dots, mu, mu_tb, mu_birth, dots_coverage, treatment_start_time):
    # Model equations (same as before)
    dSdt = mu_birth - mu * y[0] - beta * y[0] * y[2]
    dEdt = beta * y[0] * y[2] - (sigma + mu) * y[1]
    dIdt = sigma * y[1] - (get_gamma(t, gamma_no_dots, gamma_dots, treatment_start_time) + mu + mu_tb) * y[2]
    dRdt = (1 - dots_coverage) * get_gamma(t, gamma_no_dots, gamma_dots, treatment_start_time) * y[2] - mu * y[3]
    dDdt = dots_coverage * mu_tb * y[2]
    
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def get_gamma(t, gamma_no_dots, gamma_dots, treatment_start_time):
    # Time-dependent recovery rate function
    return gamma_no_dots if t < treatment_start_time else gamma_dots

# Initial conditions
S0 = 1024  # Initial number of susceptible individuals
E0 = 20    # Initial number of exposed individuals
I0 = 4     # Initial number of infectious individuals
R0 = 0     # Initial number of recovered individuals
D0 = 0     # Initial number of deaths

# Initial conditions vector
y0 = [S0, E0, I0, R0, D0]


# Parameters
params = {
    "beta": 0.58,                # Contact rate (rate of transmission)
    "sigma": 0.143,              # Rate of progression from exposed to infectious
    "gamma_no_dots": 0.167,      # Recovery rate without DOTS
    "gamma_dots": 0.74,          # Recovery rate with DOTS
    "mu": 0.0056,                # Natural death rate
    "mu_tb": 0.14,               # Death rate due to TB
    "mu_birth": 0.035,           # Birth rate
    "dots_coverage": 0.6,        # DOTS coverage (percentage of infectious individuals receiving DOTS treatment)
    "treatment_start_time": 4,   # Time in months when DOTS treatment starts
}

# Time vector
months = 12
t = np.linspace(0, months, num=30*months)  # Time span of 1 year with monthly time steps


# Solve the SEIRD model using odeint
result = odeint(seird_model, y0, t, args=tuple(params.values()))
S, E, I, R, D = result.T

# Define the parameter ranges for sensitivity analysis
problem = {
    'num_vars': len(params),
    'names': list(params.keys()),
    'bounds': [[0.5, 1.5] for _ in range(len(params))]  # Vary parameters within ±50%
}

# Generate parameter samples using Sobol's sampling
param_samples = saltelli.sample(problem, 124)  # Sample size should be a power of 2

# Initialize an array to store the results
results = []

# Perform simulations and store the results
for params_sample in param_samples:
    result = odeint(seird_model, y0, t, args=tuple(params_sample))
    results.append(result[:, 2])  # Store only the Infectious (I) compartment

# Convert results to numpy array
results_array = np.array(results)

# Perform sensitivity analysis using Sobol' method
Si = sobol.analyze(problem, results_array, print_to_console=True)

# Plot sensitivity indices
plt.figure(figsize=(10, 6))
plt.bar(range(len(params)), Si['S1'], tick_label=list(params.keys()), color='skyblue')
plt.xlabel('Parameter')
plt.ylabel('Sensitivity Index')
plt.title('Sensitivity Analysis for SEIRD Model')
plt.grid(True)
plt.show()