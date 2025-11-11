import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import K_H, kB, T

positions = load_trajectory("trajectory.txt")
positions_dt = load_trajectory("trajectory_dt.txt")

# Compute normalized histogram
hist, bins = np.histogram(positions[0], bins=200, density=True)
hist_dt, bins = np.histogram(positions_dt[0], bins=bins, density=True)
bin_centers = 0.5*(bins[:-1] + bins[1:])

# Plot stationary distribution
plt.plot(bin_centers, hist, label=r'Simulation $\Delta t = 0.1\, \tau_{BD}$')
plt.plot(bin_centers, hist_dt, label=r'Simulation $\Delta t = 0.01\, \tau_{BD}$')
plt.plot(bin_centers, np.sqrt(K_H/(2*np.pi*kB*T)) * np.exp(-K_H*bin_centers**2/(2*kB*T)), 
         'r--', label='Analytical')
plt.xlabel(r"$x/\sigma$")
plt.ylabel(r"$\rho$")
plt.legend()
plt.show()