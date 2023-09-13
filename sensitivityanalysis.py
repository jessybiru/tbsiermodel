import numpy as np
from scipy.integrate import odeint
import chaospy as cp
import matplotlib.pyplot as plt


# Define your SEIRD model and other code as before
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


# Define the parameters you want to analyze
problem = {
    'num_vars': 9,
    'names': ['beta', 'sigma', 'gamma_no_dots', 'gamma_dots', 'mu', 'mu_tb', 'mu_birth', 'dots_coverage', 'treatment_start_time'],
    'bounds': [
        [0.1, 1.0],  # beta
        [0.01, 0.5],  # sigma
        [0.1, 0.5],  # gamma_no_dots
        [0.5, 1.0],  # gamma_dots
        [0.001, 0.01],  # mu
        [0.1, 0.2],  # mu_tb
        [0.01, 0.05],  # mu_birth
        [0.5, 0.9],  # dots_coverage
        [2, 10]  # treatment_start_time
    ]
}

# Create a joint distribution for the parameters
param_dist = cp.J(*[cp.Uniform(param_min, param_max) for param_min, param_max in problem['bounds']])

# Generate parameter samples using Sobol sequence
param_samples = param_dist.sample(size=1000, rule="S")


# Initialize arrays to store model outputs for each parameter set
# Initialize arrays to store model outputs for each parameter set
outputs = np.empty((param_samples.shape[1], len(t), 5))

#outputs = np.empty((param_samples.shape[1], len(t), 5))
# Perform simulations for each parameter set
for i, params in enumerate(param_samples.T):
    result = odeint(seird_model, y0, t, args=tuple(params))
    for j in range(5):
        outputs[i, :, j] = result[:, j]  # Assign each compartment separately

# Perform simulations for each parameter set
#for i, params in enumerate(param_samples.T):
    #result = odeint(seird_model, y0, t, args=tuple(params))
    #outputs[i] = result.T

# Calculate the sensitivity indices using Sobol indices
S = cp.Sens_t(param_samples, outputs)

# Print the first-order and total-order sensitivity indices
param_names = ['beta', 'sigma', 'gamma_no_dots', 'gamma_dots', 'mu', 'mu_tb', 'mu_birth', 'dots_coverage', 'treatment_start_time']
for name, s in zip(param_names, S):
    print(f"{name}: {s}")

# Calculate the mean sensitivity indices for each parameter
mean_sensitivity = np.mean(S, axis=0)

# Plot sensitivity indices
plt.figure(figsize=(10, 6))
plt.bar(range(len(param_names)), mean_sensitivity, tick_label=param_names)
plt.xlabel('Parameter')
plt.ylabel('Mean Sensitivity Index')
plt.title('Sensitivity Analysis of SEIRD Model')
plt.grid(True)
plt.show()
