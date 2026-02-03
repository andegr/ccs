#%%
import numpy as np

from SaveToFile import load_runs, save_cluster_results
from observable_calculations import cluster_stats_traj
from SaveToFile import load_runs


#%%
# --------- Task 4) b) & c) --------- #
# --------- Cluster Analysis --------- #

sigma = 1
area_frac = 0.3

n_particles_arr = np.array([250, 350, 500])
v0_arr = np.array([0, 5, 10, 20, 35, 50])
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
    for n_particles in n_particles_arr:

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

            smax_vals[v0, n_particles, r_cut_clus]   = np.mean(smax_arr)
            nclus_vals[v0, n_particles, r_cut_clus]  = np.mean(nclus_arr)
            nmono_vals[v0, n_particles, r_cut_clus]  = np.mean(n_monomers_arr)
            smeanw_vals[v0, n_particles, r_cut_clus] = np.mean(smeanw_arr)


# %%
#-----------save results to file ------------------

save_cluster_results(
    "cluster_results_nsweep.txt",
    smax_vals,
    nclus_vals,
    nmono_vals,
    smeanw_vals,
    v0_arr,
    n_particles_arr,
    r_cut_list,
    t_sim=t_sim,
    t_eq=t_eq,
    dt=dt,
    Dt=Dt,
    Dr=Dr,
    n_runs=n_runs,
    area_frac=area_frac,
)
