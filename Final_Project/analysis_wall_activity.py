#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import one_particle_density, average_one_particle_density
from SaveToFile import load_runs
from Plot import set_Plot_Font, apply_style
from time import time
import os
# set_Plot_Font()
apply_style()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")

#%%
#------theory-------

def box_density(x, L):
    return np.where((x > 0) & (x < L), 1/L, 0.0)

def peclet(v0, Dt, Dr):
    return v0**2  / (2 * Dt * Dr)

def Deff( v0, Dt, Dr):
    return Dt * ( 1 + peclet(v0, Dt, Dr))

#%%
# ------ Loading ------- #


L = 20
t_sim = 600
t_eq = 150
dt = 0.001
n_save = 1000
n_particles = 400
Dt = 1
n_runs = 1

v0_arr = np.array([20, 10, 5, 0])   # 20, 15,  
Dr_arr = np.array([1,10])   # [1, 10]
Dt_arr = np.array([1, 10])  # [1, 10]

rho_x_dict = {}

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
                                    walls = True,
                                    pairwise = False,
                                    eta = 0,)

            bin_centers, rho_x = average_one_particle_density(
            traj_list,
            n_bins=300,
            x_range= (-2, 22.0),    # only allow bins in this range and fix therefore for each run
            direction = "x",
            )

            rho_x_dict[v0, Dt, Dr] = rho_x

#%%

fig, ax = plt.subplots()

for v0 in v0_arr:
    ax.plot(bin_centers, rho_x_dict[v0, 1, 1], linestyle="-", label=rf"$v_0$ = {v0}")
# ax.plot(bin_centers, box_density(bin_centers,L), linestyle="--", label=r"theory")

# ax.axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")


ax.set_xlabel(r"$x$ / $\sigma$ ")
ax.set_ylabel(r"$\rho(x)$")
ax.set_xlim((-2,L+2))
# ax.set_ylim((0,0.1))
ax.grid(which='both', axis='both')
ax.legend(loc="best")
# plt.show()
plt.savefig(os.path.join(plots_path, "wall_activity_dist.png"), dpi=150)



# %%
# find maxima of densities

max_dict = {}

for v0 in v0_arr:
    for Dt in Dt_arr:
        for Dr in Dr_arr:
            rho = rho_x_dict[v0, Dt, Dr]

            left_max  = np.max(rho[bin_centers < 10])
            right_max = np.max(rho[bin_centers > 10])

            max_dict[v0, Dt, Dr] = (left_max + right_max) / 2 


#%%

plot_configs = [
    (1,  1,  "blue"),
    (1, 10,  "red"),
    (10, 10, "orange"),
    (10,  1, "green"),
]

fig, ax = plt.subplots()

for Dt, Dr, color in plot_configs:
    y_vals = []

    for v0 in v0_arr:
        y_vals.append(max_dict[v0, Dt, Dr])

    ax.scatter(
        v0_arr,
        y_vals,
        color=color,
        label=rf"$D_t={Dt}$, $D_r={Dr}$, averaged maximum height at wall"
    )

ax.axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")  # (2)

ax.set_xlabel(r"$v_0 \tau_{BD} / \sigma$")
ax.set_ylabel(r"$\rho^*$")

ax.grid(which="both", axis="both")
ax.legend(loc="best")

plt.savefig(os.path.join(plots_path, "wall_activity_max.pdf"))


# %%
# find plateau height

plateau_dict = {}

for v0 in v0_arr:
    for Dt in Dt_arr:
        for Dr in Dr_arr:
            rho = rho_x_dict[v0, Dt, Dr]

            mask = (bin_centers > 5) & (bin_centers < 15)

            plateau_dict[v0, Dt, Dr] = np.mean(rho[mask])

plot_configs = [
    (1,  1,  "blue"),
    (1, 10,  "red"),
    (10, 10, "orange"),
    (10,  1, "green"),
]


fig, ax = plt.subplots()

for Dt, Dr, color in plot_configs:
    y_vals = []

    for v0 in v0_arr:
        y_vals.append(plateau_dict[v0, Dt, Dr])

    ax.scatter(
        v0_arr,
        y_vals,
        color=color,
        label=rf"$D_t={Dt}$, $D_r={Dr}$, mean plateau height"
    )

ax.axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")  # (2)

ax.set_xlabel(r"$v_0 \tau_{BD} / \sigma$")
ax.set_ylabel(r"$\bar{\rho}$")

ax.grid(which="both", axis="both")
ax.legend(loc="best")

ax.set_xscale("log")

plt.savefig(os.path.join(plots_path, "wall_activity_plateau.pdf"))


# %%
