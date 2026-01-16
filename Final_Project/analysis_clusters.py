#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from SaveToFile import load_orientations_txt, load_positions_txt
from observable_calculations import cluster_stats_traj
from Plot import set_Plot_Font
from time import time
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")

#%%
# --------- Task 4) b) & c) --------- #
# --------- Cluster Analysis --------- #

n_particles = 250
sigma = 1
area_frac = 0.3
L = np.sqrt( n_particles * np.pi * sigma**2 / (4*area_frac))

v0_arr = np.array([0, 5, 10, 20, 35, 50])
t_sim = 200
t_eq = 20
dt = 0.0001
n_save = 100
Dt = 1
Dr = 1
n_runs = 1

r_cut_clus = 1 * 2**(1/6)    # 1.5 times the normal cut_off distance

root_fname = "outputs/traj_positions_pw_n250_tsim20_teq20_dt0.0001_L25.6_v00_Dt1_Dr1_eta0.3_run0.txt"

n_steps_saved = parameters.n_steps_saved
dt_saved = parameters.dt * parameters.n_save

# initialize lists of avearage values of: 
# max_cluster_size, n_clusters, mean_mass_weighted
smax_avgs, nclus_avgs, smeanw_avgs = [], [], []         

for v0 in v0_arr:
    fname = root_fname.replace("_v00", f"_v0{v0}", 1)
    positions = load_positions_txt(filename=fname)
    smax_arr, nclus_arr, smeanw_arr = cluster_stats_traj(positions, L, r_cut_clus)

    smax_avgs.append([np.mean(smax_arr)]) 
    nclus_avgs.append([np.mean(nclus_arr)])
    smeanw_avgs.append([np.mean(smeanw_arr)])


# %%

t_arr = np.arange(0, n_steps_saved, dt_saved)

#------plotting fit----------------
fig, ax = plt.subplots()

ax.scatter(v0_arr, smax_avgs, s=15)

ax.set_xlabel(r"$v_0$]")
ax.set_ylabel(r"average Maximum Cluster $<s_\text{max}(t)>$")
# ax.set_ylim(5e-2, 1.1)   # Beispiel
ax.legend()
plt.show()
# plt.savefig(os.path.join(plots_path, "abc.pdf"))

