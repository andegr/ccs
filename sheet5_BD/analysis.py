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
from scipy.optimize import curve_fit

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

# only one particle, only one dimensio; extract that from
# the positions array, so that the autocorrelation function
# can handle it.
xs = positions[0,0,:]


acorr = autocorrelation(xs, min_sample_size=10e4)

#%%

t = np.arange(0, 0.01*len(acorr), 0.01)
# split fitting to extract both relaxation times
def monoexp(t, a, tau):
    return a * np.exp( - t/tau)

mask_fast = t < 0.1
mask_slow = t > 5

popt, pcov = curve_fit(monoexp, t[mask_fast], acorr[mask_fast])
a1, tau_fast = popt

popt, pcov = curve_fit(monoexp, t[mask_slow], acorr[mask_slow])
a2, tau_slow = popt


print("Fit parameters:")
print(f"a1={a1:.4g}, tau1={tau_fast:.2f}, a2={a2:.4g}, tau2={tau_slow:.4g}")

t_plot = t #np.arange(0, 10, 0.001)
# --- PLOT ---
fig, ax = plt.subplots()

ax.plot(t, acorr, label=r"Simulation")
ax.plot(t_plot, monoexp(t_plot, a1, tau_fast), '--', alpha=.4, label=r"Fast fit")
ax.plot(t_plot, monoexp(t_plot, a2, tau_slow), '--', alpha=.6, label=r"Slow- fit")
ax.set_xlabel(r"$t \,/\, \tau_B$")
ax.set_ylabel(r"$C_{xx}(t)$")
ax.legend(loc="best")
ax.set_ylim(0.1, 1.1)
ax.semilogy()
plt.tight_layout()
# plt.savefig("autocorr.pdf")
plt.show()

#%%
