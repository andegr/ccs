import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_timesteps_and_observable
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import x_R, n_particles

positions = load_trajectory("trajectory.txt")

# Compute normalized histogram
hist, bins = np.histogram(positions[0], bins=100, density=True)

hist *= n_particles # normalize it to Number of particles

bin_centers = 0.5*(bins[:-1] + bins[1:])

def one_body_density(x):
    rho0 = 2 * n_particles / ( x_R)
    if x < 0:
        return 0
    elif x > x_R:
        return 0
    else:
        return rho0 - rho0/x_R * x


analytical = np.copy(hist)
for i in range(len(bin_centers)):
    analytical[i] = one_body_density(bin_centers[i])

# Plot stationary distribution
plt.plot(bin_centers, hist, "g", label=r'Simulation $\Delta t = 0.001\, \tau_{BD}$')
plt.plot(bin_centers, analytical, 
         'r--', label='Analytical')
plt.xlabel(r"$x/\sigma$")
plt.ylabel(r"$\rho$")
plt.legend()
plt.show()
plt.savefig(os.path.join("plots", "stationary_distribution.pdf"))