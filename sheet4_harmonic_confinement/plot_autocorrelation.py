import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_timesteps_and_observable
from autocorrelation import autocorrelation
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import K_H, kB, T, n_save, friction_coef
from parameters import dt as dt_sim

min_sample_size = 5000
dt = dt_sim * n_save



positions_OU = load_trajectory("trajectory_OU.txt")
positions_EM = load_trajectory("trajectory.txt")

x_OU = positions_OU[0,0,:] # only one particle
x_EM = positions_EM[0,0,:] # only one particle

autocorr_OU = autocorrelation(x_OU, min_sample_size=min_sample_size)
autocorr_EM = autocorrelation(x_EM, min_sample_size=min_sample_size)

t = np.arange(0, len(autocorr_OU) * dt, dt)

autocorr_theo = np.exp( - K_H/friction_coef * t)



from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax = plt.subplots(figsize=(7, 5))

# Main plot
ax.plot(t, autocorr_OU, label="OU-Simulation")
ax.plot(t, autocorr_EM, label="EM-Simulation")
ax.plot(t, autocorr_theo, linestyle="dashed", label="Theory")
ax.set_xlabel(r"$t\,/\,\tau_{BD}$")
ax.set_ylabel(r"$C_{xx}$")
ax.set_xlim(left=-0.1)


# Inset (zoom from 0 to 3)
ax_inset = inset_axes(ax, width="45%", height="45%", loc="upper right")
ax_inset.plot(t, autocorr_OU)
ax_inset.plot(t, autocorr_EM)
ax_inset.plot(t, autocorr_theo, linestyle="dashed")
ax_inset.set_xlim(-0.1, 1)



# Optional: match y-limits automatically to the zoom region
mask = (t >= 0) & (t <= 1)
ax_inset.set_ylim(autocorr_OU[mask].min(), autocorr_OU[mask].max())
ax.legend(loc="upper left")

plt.show()





# plt.plot( t, autocorr, label="Simulation")
# plt.plot( t, autocorr_theo, linestyle = "dashed", label="Theory")
# plt.legend()
# plt.xlabel(r"$t\,/\,\tau_{BD}$")
# plt.ylabel(r"$C_{xx}$")
# plt.xlim(left=0)
# # plt.yscale("log")
# plt.show()