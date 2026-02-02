#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import calculate_average_msd
from SaveToFile import load_runs
from Plot import set_Plot_Font, apply_style
import os
# set_Plot_Font()
apply_style()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")


#%%
# --------- Task 2a a) --------- #
# --------- MSD Calculation Validation --------- #

# n_particles = 500
# t_sim = 100
# t_eq = 20
# dt = 0.001
# v0 = 0
# Dt = 1
# Dr = 1
# n_runs = 3 
# L = 10


# time_arr, traj_list = load_runs(n_particles, t_sim, t_eq, dt, L, v0, Dt, Dr, n_runs)
# mean_msd = calculate_average_msd(traj_list)

# msd_simple_theory = lambda t, Dt: 4 * Dt * t
# MSD_theo = msd_simple_theory(time_arr, Dt)

# # --- PLOTTING ---
# fig, ax = plt.subplots()

# # # Plot all EM autocorrelations
# ax.plot(time_arr, mean_msd, label=rf"Averaged MSD, {n_runs} runs")
# ax.plot(time_arr, MSD_theo, linestyle="--", label=r"MSD theory: $4 \, D_t \, t$")


# ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
# ax.set_ylabel(r"$MSD(t)$")
# # ax.set_xlim((0,10000))
# ax.legend(loc="best")
# plt.show()
# # plt.savefig(os.path.join(plots_path, "2a_a_MSD.pdf"))


#%%
#-------------------------------------------------------------------------------
#-----------asymptotic behaviour of MSD-----------------------------------------
#-------------------------------------------------------------------------------


n_particles = 500
t_sim = 800
t_eq = 1
dt = 0.001
v0 = 10
L = 10
Dt = 5
Dr = 0.1
n_runs = 3
walls = False
pairwise = False
eta = 0       

#------- simulation data --------------------------------

time_arr, traj_list = load_runs(n_particles, t_sim, t_eq, dt, L, v0, 
                                Dt, Dr, n_runs, walls, pairwise, eta)
mean_msd = calculate_average_msd(traj_list)

#%%
#------- theory --------------------------------

time = np.linspace(1e-2, 8e2, num=10000)

D_eff = Dt + v0**2 / (2*Dr)
msd_diffusive = lambda t, D_eff: 4 * D_eff * t

MSD_diffusive = msd_diffusive(time, D_eff) 

msd_ballistic = lambda t, v0, D_t: v0**2 * t**2
MSD_ballistic = msd_ballistic(time, v0, Dt)


msd_simple_theory = lambda t, D_t: 4 * D_t * t
MSD_simple_theory = msd_diffusive(time, Dt)

# %%
#------- plotting --------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.tick_params()

ax.grid(which='both', axis='both', alpha = 0.25)
# ax.plot(time_arr, MSD_simple_theory, label=r"MSD simple: $4 \, D_t \, t$")
ax.plot(time_arr, mean_msd, color="blue", label=r"$\text{MSD}(t)$ numerical")
ax.plot(time[80:], MSD_diffusive[80:], color='orange',  linestyle="--", label=r"MSD(t)$\sim 4 \, D_\text{eff} \, t$")
ax.plot(time[1:500], MSD_ballistic[1:500], color="green", linestyle="dashed", label=r"MSD(t)$\sim v_0^2 \, t^2$")
ax.plot(time[0:6], MSD_simple_theory[0:6], color="red", linestyle="dashed", label=r"MSD(t)$\sim 4 \, D_\text{t} \, t$")
ax.set_xlabel(r"$t$ / $\tau_\text{BD}$")
ax.set_ylabel(r"$\text{MSD}(t)$")
ax.set_xlim((0.01,800))
# ax.set_ylim((0.01,200000))
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc='best')
# plt.show()
plt.savefig(os.path.join(plots_path, "MSD_asymptotic_behaviour.png"), dpi=150)

# %%
