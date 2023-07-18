import pandas as pd
import numpy as np
import pylab as pl

muu= 0.5
sigma= 1/7  #exposure
gamma = 1/6 #reccovery
beta = 5 ### This was adjust to based on R0=2
## If Beta=0.53, was based on R0=3.5 (average)
## If Beta=0.85, was based on R0=5 (maximum)
nmonths = 24
dt = 0.01
npts = int(nmonths/dt)  #points to stimulate - 12 months, with dt=.01
I0 = 0.01

x = np.arange(npts)
S = np.zeros(npts)
E = np.zeros(npts)
I = np.zeros(npts)
R = np.zeros(npts)

S[0] = 1 - I0
I[0] = I0

# Vaccination parameters
#vaccination_start_month = 1  # Month when vaccination starts
#vaccination_rate = 0.02  # Adjust the vaccination rate
#
#The variable vaccination_start_month represents the month when the vaccination campaign starts.
#After this month, the susceptible population (dS) is reduced by a rate of vaccination_rate * S[t].
#


# Social distancing parameters
social_distancing_start_month = 2  # Month when social distancing measures start
social_distancing_effectiveness = 0.07  # Adjust the effectiveness of social distancing

#
#The variable social_distancing_start_month represents the month when social distancing measures are implemented.
#After this month, the transmission rate (beta) is reduced by a factor of (1 - social_distancing_effectiveness).
#
for t in x[:-1]:
    dS = -beta*S[t]*I[t] + muu*R[t]
    dE = beta*S[t]*I[t]-sigma*E[t]
    dI = sigma*E[t]-gamma*I[t]
    dR = gamma*I[t]- muu*R[t]

# Incorporate vaccination intervention
if t >= vaccination_start_month:
   dS -= vaccination_rate * S[t]

# Incorporate social distancing intervention
if t >= social_distancing_start_month:
   beta *= (1 - social_distancing_effectiveness)




S[t+1] = S[t] + dS*dt
E[t+1] = E[t] + dE*dt
I[t+1] = I[t] + dI*dt
R[t+1] = R[t] + dR*dt

time = x*dt
pl.plot(time, S, label='Susceptible')
pl.plot(time, E, label='Exposed')
pl.plot(time, I, label='Infectious')
pl.plot(time, R, label='Recovered')
pl.legend()
pl.show()