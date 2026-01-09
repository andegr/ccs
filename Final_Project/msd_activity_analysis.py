#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import calculate_average_msd
from SaveToFile import load_runs
from Plot import set_Plot_Font
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")


#%%
# --------- Task 2 b) --------- #
# --------- MSD weakly <--> highly active Brownian Particles --------- #

# v0 of [1, 3, 6, 10, 15, 20]

v0_arr = np.array([0, 8, 20])      # in units of sigma/tau

n_particles = 500
t_sim = 100
dt = 0.001
Dt = 1
Dr = 1
n_runs = 1 

D_eff_arr = Dt +  v0_arr**2 / (2*Dr)        # in units of sigma**2/tau_BD


MSDs_numerical = {}

for v0 in v0_arr:
    t, traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs)
    mean_msd = calculate_average_msd(traj_list)

    MSDs_numerical[v0] = mean_msd


msd_diffusive = lambda t, D_eff: 4 * D_eff * t


fig, ax = plt.subplots()

for D_eff, v0 in zip(D_eff_arr, v0_arr):
    line_sim, = ax.plot(t, MSDs_numerical[v0], 
                        label=r"$D_\text{eff}$" + f"={D_eff}, " \
                        + "$v_0$" + f"={v0}")
    c = line_sim.get_color()

    # theory (same color as sim, but no legend entry)
    MSD_diffusive = msd_diffusive(t, D_eff) 
    ax.plot(t, MSD_diffusive, linestyle="--", color=c, label="_nolegend_")

ax.set_xlabel(r"$t$ [$\tau_\mathrm{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
ax.set_xlim((0, 100))
# ax.set_ylim(5e-2, 1.1)   # Beispiel
ax.legend()
plt.savefig(os.path.join(plots_path, "2b_a_MSD_.pdf"))

# %%

