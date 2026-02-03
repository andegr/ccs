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
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "wall_activity_dist.png"), dpi=400)



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

fig, ax = plt.subplots(1,2,figsize=(7, 3.5),sharey=False, gridspec_kw={"wspace": 0.2},)

for Dt, Dr, color in plot_configs:
    y_vals = []

    for v0 in v0_arr:
        y_vals.append(max_dict[v0, Dt, Dr])

    ax[0].scatter(
        v0_arr,
        y_vals,
        color=color,
        label=rf"$D_t={Dt}$, $D_r={Dr}$"
    )

ax[0].axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")  # (2)

ax[0].set_xlabel(r"$v_0 \tau_{BD} / \sigma$")
ax[0].set_ylabel(r"$\rho^*$")

ax[0].grid(which="both", axis="both")


peclet_array = v0_arr**2 / (2 * Dr * Dt)

for Dt, Dr, color in plot_configs:
    y_vals = []

    for v0 in v0_arr:
        y_vals.append(max_dict[v0, Dt, Dr])

    ax[1].scatter(
        v0_arr**2 / (2 * Dr * Dt),
        y_vals,
        color=color,
        label=rf"$D_t={Dt}$, $D_r={Dr}$"
    )

ax[1].axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")  # (2)

ax[1].set_xlabel(r"$Pe_\mathrm{eff}$")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].grid(which="both", axis="both")

handles, labels = ax[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.9, 0.5),
    frameon=False,
)
plt.subplots_adjust(right=0.9)
plt.savefig(
    os.path.join(plots_path, "wall_activity_max.png"),
    dpi=400,
    bbox_inches="tight",
)


# %%
# find plateau height

plateau_dict = {}

for v0 in v0_arr:
    for Dt in Dt_arr:
        for Dr in Dr_arr:
            rho = rho_x_dict[v0, Dt, Dr]

            mask = (bin_centers > 5) & (bin_centers < 15)

            plateau_dict[v0, Dt, Dr] = np.mean(rho[mask])


#%%

# plot_configs = [
#     (1,  1,  "blue"),
#     (1, 10,  "red"),
#     (10, 10, "orange"),
#     (10,  1, "green"),
# ]


# fig, ax = plt.subplots()

# for Dt, Dr, color in plot_configs:
#     y_vals = []

#     for v0 in v0_arr:
#         y_vals.append(plateau_dict[v0, Dt, Dr])

#     ax.scatter(
#         v0_arr,
#         y_vals,
#         color=color,
#         label=rf"$D_t={Dt}$, $D_r={Dr}$"
#     )

# ax.axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")  # (2)

# ax.set_xlabel(r"$v_0 \tau_{BD} / \sigma$")
# ax.set_ylabel(r"$\bar{\rho}$")

# ax.grid(which="both", axis="both")
# ax.legend(loc="best")

# ax.set_xscale("log")

# plt.savefig(os.path.join(plots_path, "wall_activity_plateau.png"), dpi=400)


# %%

fig, ax = plt.subplots(
    1, 2,
    sharey=True,
    figsize=(7, 3.5),
    gridspec_kw={"wspace": 0.15},
)

# --------------------------
# left panel: v0 on x-axis
# --------------------------
for Dt, Dr, color in plot_configs:
    y_vals = [plateau_dict[v0, Dt, Dr] for v0 in v0_arr]
    ax[0].scatter(
        v0_arr,
        y_vals,
        color=color,
        label=rf"$D_t={Dt}$, $D_r={Dr}$"
    )

ax[0].axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")
ax[0].set_xlabel(r"$v_0 \tau_{BD} / \sigma$")
ax[0].set_ylabel(r"$\bar{\rho}$")
ax[0].set_xscale("log")
ax[0].grid(which="both", axis="both")

# --------------------------
# right panel: Pe on x-axis
# --------------------------
for Dt, Dr, color in plot_configs:
    y_vals = [plateau_dict[v0, Dt, Dr] for v0 in v0_arr]
    peclet_array = v0_arr**2 / (2 * Dr * Dt)
    ax[1].scatter(
        peclet_array,
        y_vals,
        color=color,
        label=rf"$D_t={Dt}$, $D_r={Dr}$"
    )

ax[1].axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")
ax[1].set_xlabel(r"$Pe_\mathrm{eff}$")
ax[1].set_xscale("log")
ax[1].grid(which="both", axis="both")

# --------------------------
# shared legend outside
# --------------------------
handles, labels = ax[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.9, 0.5),
    frameon=False,
)

plt.subplots_adjust(right=0.9)

plt.savefig(
    os.path.join(plots_path, "wall_activity_plateau.png"),
    dpi=400,
    bbox_inches="tight",
)

# %%
