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

min_sample_size = 50
dt = dt_sim * n_save

positions1 = load_trajectory("trajectory.txt")
positions01 = load_trajectory("trajectory01.txt")
positions001 = load_trajectory("trajectory01.txt")
positions0001 = load_trajectory("trajectory01.txt")

x1 = positions1[0,0,:] # only one particle
x01 = positions01[0,0,:] # only one particle
x001 = positions001[0,0,:] # only one particle
x0001 = positions0001[0,0,:] # only one particle

autocorr1 = autocorrelation(x1, min_sample_size=min_sample_size)
autocorr01 = autocorrelation(x01, min_sample_size=min_sample_size)
autocorr001 = autocorrelation(x001, min_sample_size=min_sample_size)
autocorr0001 = autocorrelation(x0001, min_sample_size=min_sample_size)

t = np.arange(0, len(autocorr1) * dt, dt)

autocorr_theo = np.exp( - K_H/friction_coef * t)


# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(7, 5))

# Plot all EM autocorrelations
ax.plot(t, autocorr1,     label=r"EM $\Delta t = 0.1\,\tau_{BD}$")
ax.plot(t, autocorr01,    label=r"EM $\Delta t = 0.01\,\tau_{BD}$")
ax.plot(t, autocorr001,   label=r"EM $\Delta t = 0.001\,\tau_{BD}$")
ax.plot(t, autocorr0001,  label=r"EM $\Delta t = 0.0001\,\tau_{BD}$")

# Plot theory
ax.plot(t, autocorr_theo, linestyle="dashed", label="Theory")

ax.set_xlabel(r"$t\,/\,\tau_{BD}$")
ax.set_ylabel(r"$C_{xx}$")
ax.set_xlim(left=-0.1)
ax.legend(loc="upper right")


# --- INSET ---
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax, width="45%", height="45%", loc="upper left")

ax_inset.plot(t, autocorr1)
ax_inset.plot(t, autocorr01)
ax_inset.plot(t, autocorr001)
ax_inset.plot(t, autocorr0001)
ax_inset.plot(t, autocorr_theo, linestyle="dashed")

# zoom region: t = 0 ... 1
mask = (t >= 0) & (t <= 1)
ax_inset.set_xlim(0, 1)

# auto-set y-limits based on a reference curve
y_min = min(autocorr1[mask].min(),
            autocorr01[mask].min(),
            autocorr001[mask].min(),
            autocorr0001[mask].min())
y_max = max(autocorr1[mask].max(),
            autocorr01[mask].max(),
            autocorr001[mask].max(),
            autocorr0001[mask].max())

ax_inset.set_ylim(y_min, y_max)

plt.show()




# positions_OU = load_trajectory("trajectory_OU.txt")
# positions_EM = load_trajectory("trajectory.txt")

# x_OU = positions_OU[0,0,:] # only one particle
# x_EM = positions_EM[0,0,:] # only one particle

# autocorr_OU = autocorrelation(x_OU, min_sample_size=min_sample_size)
# autocorr_EM = autocorrelation(x_EM, min_sample_size=min_sample_size)

# t = np.arange(0, len(autocorr_OU) * dt, dt)

# autocorr_theo = np.exp( - K_H/friction_coef * t)



# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# fig, ax = plt.subplots(figsize=(7, 5))

# # Main plot
# ax.plot(t, autocorr_OU, label="OU-Simulation")
# ax.plot(t, autocorr_EM, label="EM-Simulation")
# ax.plot(t, autocorr_theo, linestyle="dashed", label="Theory")
# ax.set_xlabel(r"$t\,/\,\tau_{BD}$")
# ax.set_ylabel(r"$C_{xx}$")
# ax.set_xlim(left=-0.1)


# # Inset (zoom from 0 to 3)
# ax_inset = inset_axes(ax, width="45%", height="45%", loc="upper right")
# ax_inset.plot(t, autocorr_OU)
# ax_inset.plot(t, autocorr_EM)
# ax_inset.plot(t, autocorr_theo, linestyle="dashed")
# ax_inset.set_xlim(-0.1, 1)



# # Optional: match y-limits automatically to the zoom region
# mask = (t >= 0) & (t <= 1)
# ax_inset.set_ylim(autocorr_OU[mask].min(), autocorr_OU[mask].max())
# ax.legend(loc="upper left")

# plt.show()