#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from SaveToFile import load_runs
from observable_calculations import cluster_stats_traj
from Plot import apply_style
from time import time
import os
apply_style()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")

#%%
# --------- Task 4) b) & c) --------- #
# --------- Cluster Analysis --------- #

sigma = 1
area_frac_arr = np.array([0.3, 0.4, 0.5])


n_particles = 250
v0_arr = np.array([0, 10, 20, 35, 50])
t_sim = 200
t_eq = 20
dt = 0.0001
n_save = 100
Dt = 1
Dr = 1
n_runs = 1

r_cutoff = 2**(1/6) 

r_cut_factors = [0.99, 1.0, 1.01]
r_cut_list = [f * r_cutoff for f in r_cut_factors]


# initialize lists of avearage values of: 
# max_cluster_size, n_clusters, mean_mass_weighted
smax_vals, nclus_vals, nmono_vals, smeanw_vals = {}, {}, {}, {}
        


for v0 in v0_arr:
    for area_frac in area_frac_arr:

        L = np.sqrt(n_particles * np.pi * sigma**2 / (4 * area_frac))

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
            walls=False,
            pairwise=True,
            eta=area_frac,
        )

        if n_runs != 1:
            print("Warning: Only first run is used for analysis")

        for r_cut_clus in r_cut_list:
            smax_arr, nclus_arr, n_monomers_arr, smeanw_arr = \
                cluster_stats_traj(traj_list[0], L, r_cut_clus)

            smax_vals[v0, area_frac, r_cut_clus]   = np.mean(smax_arr)
            nclus_vals[v0, area_frac, r_cut_clus]  = np.mean(nclus_arr)
            nmono_vals[v0, area_frac, r_cut_clus]  = np.mean(n_monomers_arr)
            smeanw_vals[v0, area_frac, r_cut_clus] = np.mean(smeanw_arr)


# %%
# ---------- plotting all observables ----------
n_particles = np.array([250])
colors = {
    area_frac_arr: f"C{i}"
    for i, area_frac_arr in enumerate(area_frac_arr)
}


linestyles = {
    r_cut_list[0]: "--",
    r_cut_list[1]: "-",
    r_cut_list[2]: "--",
}

alphas = {
    r_cut_list[0]: 0.3,
    r_cut_list[1]: 1,
    r_cut_list[2]: 0.3,
}


labels_rcut = {
    r_cut_list[0]: rf"${r_cut_list[0]:.2f}\,r_c$",
    r_cut_list[1]: rf"${r_cut_list[1]:.2f}\,r_c$",
    r_cut_list[2]: rf"${r_cut_list[2]:.2f}\,r_c$",
}

fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [smax_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[0].plot(
            v0_arr, y,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut], 
            marker="o",
            label=r"$\eta$" + f"={area_frac}, {labels_rcut[r_cut]}"
        )

axes[0].set_ylabel(r"$\langle s_{\max} \rangle$")
axes[0].legend(fontsize=8)
axes[0].grid(True)


# (2) Number of clusters <n_clus>
for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [nclus_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[1].plot(
            v0_arr, y/n_particles,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[1].set_ylabel(r"$\langle n_{\mathrm{clus}} / n \rangle$")
axes[1].grid(True)


# (3) Number of monomers <n_mono>
for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [nmono_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[2].plot(
            v0_arr, y/n_particles,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[2].set_ylabel(r"$\langle n_{\mathrm{mono}} / n \rangle$")
axes[2].grid(True)


# (4) Weighted mean cluster size <s_mean^w>
for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [smeanw_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[3].plot(
            v0_arr, y,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[3].set_ylabel(r"$\langle s_{\mathrm{mean}}^{(w)} \rangle$")
axes[3].set_xlabel(r"$v_0$")
axes[3].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plots_path, "cluster_analysis_areaFrac.png"))

# plt.savefig("/cluster_analysis_areaFrac.pdf")
plt.show()


# %%
# ---------- plotting all observables ----------
n_particles = np.array([250])
colors = {
    area_frac_arr: f"C{i}"
    for i, area_frac_arr in enumerate(area_frac_arr)
}


linestyles = {
    r_cut_list[0]: "--",
    r_cut_list[1]: "-",
    r_cut_list[2]: "--",
}

alphas = {
    r_cut_list[0]: 0.3,
    r_cut_list[1]: 1,
    r_cut_list[2]: 0.3,
}


labels_rcut = {
    r_cut_list[0]: rf"${r_cut_list[0]:.2f}\,r_c$",
    r_cut_list[1]: rf"${r_cut_list[1]:.2f}\,r_c$",
    r_cut_list[2]: rf"${r_cut_list[2]:.2f}\,r_c$",
}

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
# leave room on the right for the legend (do this before tight_layout or right after creating fig)
fig.subplots_adjust(right=0.78)


for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [smax_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[0].plot(
            v0_arr, y,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut], 
            marker="o",
            label=r"$\eta$" + f"={area_frac}, {labels_rcut[r_cut]}"
        )

axes[0].set_ylabel(r"$\langle s_{\max} \rangle$")
# put legend outside, middle right
axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=15)
axes[0].grid(True)


# (2) Number of clusters <n_clus>
for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [nclus_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[1].plot(
            v0_arr, y/n_particles,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[1].set_ylabel(r"$\langle n_{\mathrm{clus}} / n \rangle$")
axes[1].grid(True)


# (3) Weighted mean cluster size <s_mean^w>
for area_frac in area_frac_arr:
    for r_cut in r_cut_list:
        y = [smeanw_vals[v0, area_frac, r_cut] for v0 in v0_arr]
        axes[2].plot(
            v0_arr, y,
            color=colors[area_frac],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[2].set_ylabel(r"$\langle s_{\mathrm{mean}}^{(w)} \rangle$")
axes[2].set_xlabel(r"$v_0$")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plots_path, "cluster_analysis_areaFrac.png"), dpi=150)

# plt.savefig("/cluster_analysis_areaFrac.pdf")
plt.show()

# %%
