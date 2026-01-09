#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import msd_numerical, msd_theory
from SaveToFile import load_positions_txt, load_runs
from Plot import set_Plot_Font
from time import time
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")





#%%
# --------- Task 2a a) --------- #
# --------- MSD Calculation Validation --------- #
n_particles = 500
t_sim = 100
dt = 0.001
v0 = 0
Dt = 1
Dr = 1
n_runs = 10 



traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs)


msds = []
for traj in traj_list:
    run_msd = msd_numerical(traj)
    msds.append(run_msd)

msds = np.array(msds)
mean_msd = np.mean(msds, axis=0)

dt_saved = dt * parameters.n_save  #
time_arr = np.arange(traj.shape[2]) * dt_saved

n_steps_saved = parameters.n_steps_saved
msd_simple_theory = lambda t, Dt: 4 * Dt * t
MSD_theo = msd_simple_theory(time_arr, Dt)

# --- PLOTTING ---
fig, ax = plt.subplots()

# # Plot all EM autocorrelations
ax.plot(time_arr, mean_msd, label=rf"Averaged MSD, {n_runs} runs")
ax.plot(time_arr, MSD_theo, linestyle="--", label=r"MSD theory: $4 \, D_t \, t$")


ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
# ax.set_xlim((0,10000))
ax.legend(loc="best")
plt.show()
# plt.savefig(os.path.join(plots_path, "2a_a_MSD.pdf"))
