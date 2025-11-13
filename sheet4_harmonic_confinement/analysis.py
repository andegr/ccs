#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_timesteps_and_observable
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import K_H, kB, T

positions = load_trajectory("trajectory.txt")
positions_OU = load_trajectory("trajectory_OU.txt")
# positions_dt = load_trajectory("trajectory_dt.txt")

# Compute normalized histogram
hist, bins = np.histogram(positions[0], bins=200, density=True)
hist_OU, bins_OU = np.histogram(positions_OU[0], bins=200, density=True)
# hist_dt, bins = np.histogram(positions_dt[0], bins=bins, density=True)
bin_centers = 0.5*(bins[:-1] + bins[1:])
bin_centers_OU = 0.5*(bins_OU[:-1] + bins_OU[1:])

# Plot stationary distribution
plt.plot(bin_centers, hist, "g", label=r'Simulation $\Delta t = 0.1\, \tau_{BD}$')
plt.plot(bin_centers_OU, hist_OU, "b", label=r'Simulation wit OU, $\Delta t = 0.1\, \tau_{BD}$')
# plt.plot(bin_centers, hist_dt, label=r'Simulation $\Delta t = 0.01\, \tau_{BD}$')
plt.plot(bin_centers, np.sqrt(K_H/(2*np.pi*kB*T)) * np.exp(-K_H*bin_centers**2/(2*kB*T)), 
         'r--', label='Analytical')
plt.xlabel(r"$x/\sigma$")
plt.ylabel(r"$\rho$")
plt.legend()
# plt.show()
plt.savefig(os.path.join("plots", "1c)_stat_distr_dt0.01.pdf"))


#%%


timesteps_acorr, acorr = load_timesteps_and_observable("acorr.txt")
timesteps_acorr, acorr_OU = load_timesteps_and_observable("acorr_OU.txt")

plt.figure("1d)")

# Plot stationary distribution
plt.plot(timesteps_acorr[:200], acorr[:200], "g", label=r'$C_{xx, EM}$ for $\Delta t = 0.1\, \tau_{BD}, K_H=10$')
plt.plot(timesteps_acorr[:200], acorr_OU[:200], "b", label=r'$C_{xx, OU} $ for $\Delta t = 0.1\, \tau_{BD}, K_H=10$')
plt.legend()
plt.savefig(os.path.join("plots", "1d)_stat_distr_dt0.01_nsave1e5_KH10_tsim5e6_.pdf"))
# plt.show()




# %%
