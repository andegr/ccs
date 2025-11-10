import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_timesteps_and_observable
import os

from parameters import dt

os.chdir(os.path.dirname(__file__))

# Load displacement data
_, r_x = load_timesteps_and_observable("displ_vec_x_axis.txt")
_, r_y = load_timesteps_and_observable("displ_vec_y_axis.txt")

# Parameters
kBT = 1.0   # thermal energy
xi = 1.0    # friction coefficient

# Compute theoretical standard deviation
sigma = np.sqrt(4 * kBT * dt / xi)

# Prepare theoretical Gaussian
r_vals = np.linspace(-3*sigma, 3*sigma, 200)
P_theo = (1 / np.sqrt(4 * np.pi * kBT * dt / xi)) * np.exp(-r_vals**2 * xi / (4 * kBT * dt))

# Create figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for R_x
axs[0].hist(r_x, bins=30, density=True, alpha=0.6, label='Simulated $R_x$')
axs[0].plot(r_vals, P_theo, 'r-', lw=2, label='Analytical $P_{theo}$')
axs[0].set_xlabel('$R_x / \sigma$')
axs[0].set_ylabel('$P(R_x)$')
# axs[0].set_title('Random Displacement: $R_x$')
axs[0].legend()
axs[0].grid(True)

# Plot for R_y
axs[1].hist(r_y, bins=30, density=True, alpha=0.6, label='Simulated $R_y$')
axs[1].plot(r_vals, P_theo, 'r-', lw=2, label='Analytical $P_{theo}$')
axs[1].set_xlabel('$R_y / \sigma$')
axs[1].set_ylabel('$P(R_y)$')
# axs[1].set_title('Random Displacement: $R_y$')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

