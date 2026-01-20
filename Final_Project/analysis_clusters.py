#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from SaveToFile import load_runs
from observable_calculations import cluster_stats_traj
from Plot import set_Plot_Font
from time import time
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
# plots_path = os.path.join(os.path.dirname(__file__), "plots")

#%%
# --------- Task 4) b) & c) --------- #
# --------- Cluster Analysis --------- #

sigma = 1
area_frac = 0.3

n_particles_arr = np.array([250])
v0_arr = np.array([0, 5, 10, 20, 35, 50])
t_sim = 200
t_eq = 20
dt = 0.0001
n_save = 100
Dt = 1
Dr = 1
n_runs = 1

r_cutoff = 2**(1/6) 
r_cut_clus = 1 * r_cutoff


# initialize lists of avearage values of: 
# max_cluster_size, n_clusters, mean_mass_weighted
smax_avgs, nclus_avgs, nmono_avgs, smeanw_avgs = {}, {}, {}, {}         


for v0 in v0_arr:
    for n_particles in n_particles_arr:
        L = np.sqrt( n_particles * np.pi * sigma**2 / (4*area_frac))
        t_arr, traj_list = load_runs(
                                    n_particles,
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
                                    eta = area_frac,
                                    )
        if n_runs != 1:
            print("Warning: Only first run is used for analysis")

        smax_arr, nclus_arr, n_monomers_arr, smeanw_arr = cluster_stats_traj(traj_list[0], L, r_cut_clus)

        smax_avgs[ v0, n_particles] = np.mean(smax_arr)
        nclus_avgs[ v0, n_particles] = np.mean(nclus_arr)
        nmono_avgs[ v0, n_particles] = np.mean(n_monomers_arr)
        smeanw_avgs[ v0, n_particles] = np.mean(smeanw_arr)


# %%

# ---------- plotting all observables ----------
fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(7, 10),
    sharex=True
)

# (1) Maximum cluster size <s_max>
for n_particles in n_particles_arr:
    y = [smax_avgs[v0, n_particles] for v0 in v0_arr]
    axes[0].plot(
        v0_arr,
        y,
        marker="o",
        label=f"N = {n_particles}"
    )
axes[0].set_ylabel(r"$\langle s_{\max} \rangle$")
axes[0].legend()
axes[0].grid(True)

# (2) Number of clusters <n_clus>
for n_particles in n_particles_arr:
    y = [nclus_avgs[v0, n_particles] for v0 in v0_arr]
    axes[1].plot(v0_arr, y, marker="o")
axes[1].set_ylabel(r"$\langle n_{\mathrm{clus}} \rangle$")
axes[1].grid(True)

# (3) Number of monomers <n_mono>
for n_particles in n_particles_arr:
    y = [nmono_avgs[v0, n_particles] for v0 in v0_arr]
    axes[2].plot(v0_arr, y, marker="o")
axes[2].set_ylabel(r"$\langle n_{\mathrm{mono}} \rangle$")
axes[2].grid(True)

# (4) Weighted mean cluster size <s_mean^w>
for n_particles in n_particles_arr:
    y = [smeanw_avgs[v0, n_particles] for v0 in v0_arr]
    axes[3].plot(v0_arr, y, marker="o")
axes[3].set_ylabel(r"$\langle s_{\mathrm{mean}}^{(w)} \rangle$")
axes[3].set_xlabel(r"$v_0$")
axes[3].grid(True)

plt.tight_layout()
plt.show()

# %%
