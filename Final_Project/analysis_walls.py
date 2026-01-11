#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import one_particle_density, average_one_particle_density
from SaveToFile import load_runs
from Plot import set_Plot_Font
from time import time
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")


#%%
# ------ Loading ------- #


L = 20
v0 = 0
t_sim = 100
dt = 0.001
n_save = 100
n_particles = 250
Dt = 1
Dr = 1
n_runs = 2

t, traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs, walls=True, L=L)

bin_centers, rho_mean = average_one_particle_density(
    traj_list,
    n_bins=200,
    x_range=(0.0, 20.0)     # only allow bins in this range and fix therefore for each run
)


#%%
# ------ Plotting ------- #

fig, ax = plt.subplots()

ax.plot(bin_centers, rho_mean, linestyle="-", label=r"")
ax.set_xlabel(r"$x$ / $\sigma$ ")
ax.set_ylabel(r"$\rho(x)$")
ax.set_xlim((0,L))
ax.set_ylim((0,0.3))
ax.grid(which='both', axis='both')
ax.legend(loc="best")
plt.show()
# plt.savefig(os.path.join(plots_path, "MSD_asymptotic_behaviour.pdf"))
# %%
