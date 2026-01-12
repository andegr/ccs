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
#------theory-------

def box_density(x, L):
    return np.where((x > 0) & (x < L), 1/L, 0.0)


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
n_runs = 10


t, traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs, walls=True, L=L)

#%%
#---------- x- direction ------------

bin_centers, rho_mean = average_one_particle_density(
    traj_list,
    n_bins=300,
    x_range= (-2, 22.0),    # only allow bins in this range and fix therefore for each run
    direction = "x",
    )


#%%
# ------ Plotting ------- #

fig, ax = plt.subplots()

ax.plot(bin_centers, rho_mean, linestyle="-", label=r"simulation")
ax.plot(bin_centers, box_density(bin_centers,L), linestyle="--", label=r"theory")
ax.set_xlabel(r"$x$ / $\sigma$ ")
ax.set_ylabel(r"$\rho(x)$")
ax.set_xlim((-2,L+2))
ax.set_ylim((0,0.1))
ax.grid(which='both', axis='both')
ax.legend(loc="best")
plt.show()
# plt.savefig(os.path.join(plots_path, "MSD_asymptotic_behaviour.pdf"))
# %%


#%%
#---------- y- direction ------------

# map particles in box (PBC)
traj_list_pbc_y = []
for traj in traj_list:

    traj_list_pbc_y.append(traj % L)


bin_centers, rho_mean_unwrapped = average_one_particle_density(
    traj_list,
    n_bins=300,
    x_range= (-50, 50),    # only allow bins in this range and fix therefore for each run
    direction = "y",
    )

bin_centers, rho_mean = average_one_particle_density(
    traj_list_pbc_y,
    n_bins=300,
    x_range= (-50, 50),    # only allow bins in this range and fix therefore for each run
    direction = "y",
    )


#%%
# ------ Plotting ------- #
fig, ax = plt.subplots()

ax.plot(bin_centers, rho_mean_unwrapped, linestyle="-", label=r"unwrapped coordinates")
ax.plot(bin_centers, rho_mean, linestyle="-", label=r"coordinates with PBC")
ax.plot(bin_centers, box_density(bin_centers,L), linestyle="--", label=r"theory for PBC")


ax.set_xlabel(r"$y$ / $\sigma$ ")
ax.set_ylabel(r"$\rho(y)$")
# ax.set_xlim((-2,L+2))
# ax.set_ylim((0,0.3))
ax.grid(which='both', axis='both')
ax.legend(loc="best")
plt.show()
# plt.savefig(os.path.join(plots_path, "MSD_asymptotic_behaviour.pdf"))
# %%