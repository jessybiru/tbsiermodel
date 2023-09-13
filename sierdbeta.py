import numpy as np
from scipy.integrate import odeint
import chaospy as cp
import matplotlib.pyplot as plt

# Define your SEIRD model and other code as before
# (Include seird_model, get_gamma, initial conditions, and time vector)
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
gamma_dots = 0.74       # Recovery rate with DOTS

# Time vector
months =12
t = np.linspace(0, months, num=30*months)  # Time span of 1 year with monthly time steps

#dots coverage
dots_coverage = 0.6  # DOTS coverage (percentage of infectious individuals receiving DOTS treatment)
treatment_start_time = 4  # Time in months when DOTS treatment starts

# Initial conditions vector
y0 = [S0, E0, I0, R0, D0]

# Parameters
param_names = ['beta', 'sigma', 'gamma_dots']

# Define parameter bounds for sensitivity analysis
param_bounds = {
    'beta': [0.1, 1.0],
    'sigma': [0.01, 0.5],
    'gamma_dots': [0.5, 1.0]
}

# Create a joint distribution for the selected parameters
param_dist = cp.J(*[cp.Uniform(param_bounds[param][0], param_bounds[param][1]) for param in param_names])

# Generate parameter samples using Sobol sequence
param_samples = param_dist.sample(size=1000, rule="S")

# Initialize arrays to store model outputs for each parameter set
outputs = np.empty((len(param_names), param_samples.shape[0], len(t), 5))

# Keep the other parameters constant
constant_params = {
    'gamma_no_dots': 0.167,  # Set to the constant value
    'mu': 0.0056,             # Set to the constant value
    'mu_tb': 0.14,            # Set to the constant value
    'mu_birth': 0.035,        # Set to the constant value
    'dots_coverage': 0.6,     # Set to the constant value
    'treatment_start_time': 4  # Set to the constant value
}

# Perform simulations for each parameter set
for i, param_name in enumerate(param_names):
    for j, param_value in enumerate(param_samples.T[i]):
        # Create a dictionary with all parameters, including the selected one and constants
        full_params = {param: param_value for param in param_names}
        full_params.update(constant_params)  # Add the constant parameters
        result = odeint(seird_model, y0, t, args=tuple(full_params.values()))
        for k in range(5):
            outputs[i, j, :, k] = result[:, k]  # Assign each compartment separately

# Calculate the sensitivity indices using Sobol indices
S = cp.Sens_t(param_samples, outputs)

 #Print the first-order and total-order sensitivity indices for the selected parameters
for i, param_name in enumerate(param_names):
    first_order_sensitivity = [float(s[0]) for s in S[i]]
    total_order_sensitivity = [float(s[1]) for s in S[i]]
    print(f"{param_name}:")
    for j in range(len(param_samples)):
        print(f"  Sample {j + 1}:")
        print(f"    First-order sensitivity: {first_order_sensitivity[j]:.4f}")
        print(f"    Total-order sensitivity: {total_order_sensitivity[j]:.4f}")

# Plot sensitivity indices
plt.figure(figsize=(10, 6))
plt.bar(range(len(param_names)), S[:, 0], tick_label=param_names)
plt.xlabel('Parameter')
plt.ylabel('First-order Sensitivity Index')
plt.title('First-order Sensitivity Analysis for Selected Parameters')
plt.grid(True)
plt.show()