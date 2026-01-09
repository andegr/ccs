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
# --------- Task 2a a) --------- #
# --------- MSD Calculation Validation --------- #

n_particles = 500
t_sim = 100
dt = 0.001
v0 = 0
Dt = 1
Dr = 1
n_runs = 10 



time_arr, traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs)
mean_msd = calculate_average_msd(traj_list)

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


#%%
#-------------------------------------------------------------------------------
#-----------asymptotic behaviour of MSD-----------------------------------------
#-------------------------------------------------------------------------------


n_particles = 500
t_sim = 100
dt = 0.001
v0 = 8
Dt = 1
Dr = 1
n_runs = 10 


#------- simulation data --------------------------------

time_arr, traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs)
mean_msd = calculate_average_msd(traj_list)


#------- theory --------------------------------
D_eff = Dt + v0**2 / (2*Dr)
msd_diffusive = lambda t, D_eff: 4 * D_eff * t

MSD_diffusive = msd_diffusive(time_arr, D_eff) 

msd_ballistic = lambda t, v0, D_t: v0**2 * t**2 + 4 * D_t * t
MSD_ballistic = msd_ballistic(time_arr, v0, Dt)


msd_simple_theory = lambda t, D_t: 4 * D_t * t
MSD_simple_theory = msd_diffusive(time_arr, Dt)



#------- plotting --------------------------------

fig, ax = plt.subplots()

# ax.plot(time_arr, MSD_simple_theory, label=r"MSD simple: $4 \, D_t \, t$")
ax.plot(time_arr, mean_msd, label=r"MSD numerical")
ax.plot(time_arr, MSD_diffusive, linestyle="--", label=r"MSD diffusive: $4 \, D_\text{eff} \, t$")
ax.plot(time_arr, MSD_ballistic, linestyle="--", label=r"MSD ballistic")
ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
ax.set_xlim((0.01,100))
ax.set_ylim((0.01,5000))
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="best")
# plt.show()
plt.savefig(os.path.join(plots_path, "MSD_asymptotic_behaviour.pdf"))
# %%
