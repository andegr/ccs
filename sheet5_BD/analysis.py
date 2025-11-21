#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_positions_txt
import Plot
from parameters import A,B,C, kB, T, V_0, n_particles, n_steps_acorr
import os
import time
import logging
from autocorrelation import autocorrelation

Plot.apply_style()
#%%
x_1 = 0
x_23 = np.sqrt(B/(2*A))

x_arr = np.linspace(-3*x_23-x_23, 3*x_23-x_23, 1000)

def rho_eq_analytical(x_arr):
    prefac = 0.5 * np.sqrt(2*V_0*B/(np.pi*kB*T))
    return  prefac* (np.exp(-2*V_0/(kB*T) * (x_arr - x_23)**2) + np.exp(-2*V_0/(kB*T) * (x_arr + x_23)**2))

rho_eq_ana = rho_eq_analytical(x_arr)

# changing path to sheet 5
os.chdir(os.path.dirname(__file__))
# start_time = time.time()
# positions = load_trajectory("trajectory_OVITO_tSim1e4_tEQ500.txt")
# print(f"Finished loading OVITO trajectory with total time of {time.time() - start_time:.2f} s")

"""Loading was this much faster with other:
Loaded trajectory from trajectory_OVITO.txt
Finished loading OVITO trajectory with total time of 0.73 s
Loaded positions from trajectory.txt
Finished loading trajectory (less overhead) with total time of 0.04 s"""

start_time = time.time()
positions = load_positions_txt("trajectory.txt")
print(f"Finished loading trajectory (less overhead) with total time of {time.time() - start_time:.2f} s")

# Histogram
hist, bins = np.histogram(positions[0], bins=100, density=True) # just taking the first particle with [0]
bin_centers = 0.5*(bins[:-1] + bins[1:])

# --- PLOTTING ---
fig, ax = plt.subplots()

# # Plot all EM autocorrelations
ax.plot(x_arr, rho_eq_ana, label=r"Analytical")
ax.plot(bin_centers, hist, label=r"Data")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\rho_\text{eq}(x)$")
# ax.set_xlim(left=-0.1)
ax.legend(loc="best")
# plt.savefig()
plt.show()
# %%

print(len(positions[0]))

acorr = autocorrelation(positions[0], min_sample_size=10)
x_arr = np.linspace(-n_steps_acorr, n_steps_acorr, len(acorr))

print(x_arr)

# --- PLOTTING ---
fig, ax = plt.subplots()

ax.plot(x_arr, acorr, label=r"Simulation")
# ax.plot(bin_centers, hist, label=r"Data")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Autocorrelation $<x(t)x(t+)>$")
# ax.set_xlim(left=-0.1)
ax.legend(loc="best")
# plt.savefig()
plt.show()

# %%
