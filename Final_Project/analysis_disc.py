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


L = 30
t_sim = 100
t_eq = 25
dt = 0.0001
n_save = 1000
n_particles = 250
Dt = 1
n_runs = 1
eta = 0.3

r_disc = 15
epsilon_disc = 100

v0_arr = np.array([0, 10, 20, 35]) #np.array([50, 35, 20, 10, 0]) 
Dr_arr = np.array([1])
Dt_arr = np.array([1])

rho_r_dict = {}

for v0 in v0_arr:
    for Dr in Dr_arr:
        for Dt in Dt_arr:
            t, traj_list = load_runs(n_particles,
                                    t_sim,
                                    t_eq,
                                    dt,
                                    L,
                                    v0,
                                    Dt,
                                    Dr,
                                    n_runs,
                                    walls = False,
                                    pairwise = True,
                                    eta = eta,
                                    disc = True,
                                    r_disc = r_disc,
                                    epsilon_disc = epsilon_disc,
                                    )

            bin_centers, rho_r = average_one_particle_density(
            traj_list,
            n_bins=300,
            x_range= ( 0, 16),    # only allow bins in this range and fix therefore for each run
            direction = "radial",
            L = L,
            )

            rho_r_dict[v0, Dt, Dr] = rho_r

#%%

fig, ax = plt.subplots()

for v0 in v0_arr:
    ax.plot(bin_centers, rho_r_dict[v0, 1, 1], linestyle="-", label=rf"$v_0$ = {v0}")
# ax.plot(bin_centers, box_density(bin_centers,L), linestyle="--", label=r"theory")

# ax.axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")


ax.set_xlabel(r"$r$ / $\sigma$ ")
ax.set_ylabel(r"$\rho(r)$")
ax.set_xlim((0,r_disc +1))
# ax.set_ylim((0,0.1))
ax.grid(which='both', axis='both')
ax.legend(loc="best")
# plt.show()
plt.savefig(os.path.join(plots_path, "disc_activity_dist.pdf"))



# %%
