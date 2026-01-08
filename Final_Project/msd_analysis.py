#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import msd_numerical, msd_theory, orientation_autocorrelation
from SaveToFile import load_orientations_txt, load_positions_txt
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
# v0 = 0

fname = "outputs/traj_positions_n500_tsim100_dt0.001_v00_Dt1_Dr1.txt"
v0 = 0
Dt = 1
Dr = 1
dt = 0.001
t_sim = 100        # in units of tau_BD


start_time = time()
print("Started loading trajectory and MSD calculation...")
traj_sim = load_positions_txt(filename=fname)
MSD_numerical = msd_numerical(traj_sim)
print(f"Finished loading trajectory and MSD calculation with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save  #
time_arr = np.arange(traj_sim.shape[2]) * dt_saved

n_steps_saved = parameters.n_steps_saved
msd_simple_theory = lambda t, Dt: 4 * Dt * t
MSD_theo = msd_simple_theory(time_arr, Dt)

# --- PLOTTING ---
fig, ax = plt.subplots()

# # Plot all EM autocorrelations
ax.plot(time_arr, MSD_theo, label=r"MSD theory: $4 \, D_t \, t$")
ax.plot(time_arr, MSD_numerical, label=r"MSD numerical")

ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
# ax.set_xlim((0,10000))
ax.legend(loc="best")
# plt.show()
plt.savefig(os.path.join(plots_path, "2a_a_MSD.pdf"))

#%%
# --------- Task 2a c) --------- #
# --------- MSD Short time behaviour Validation --------- #

# v0 > 0,   Dr --> relatively big such that one can see the short time behaviour


fname = "outputs/traj_positions_n500_tsim200_dt0.001_v08_Dt1_Dr10.txt"
Dt = 1
Dr = 10
v0 = 8
dt = 0.001
t_sim = 200

start_time = time()
print("Started loading trajectory and MSD calculation...")
traj_sim = load_positions_txt(filename=fname)
MSD_numerical = msd_numerical(traj_sim)
print(f"Finished loading trajectory and MSD calculation with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save  #
time_arr = np.arange(traj_sim.shape[2]) * dt_saved

msd_ballistic = lambda t, v0: v0**2 * t**2
MSD_ballistic = msd_ballistic(time_arr, v0)

msd_linear = lambda t, Dt: 4*Dt * t
MSD_linear = msd_linear(time_arr, Dt)
#%%
# --- PLOTTING 2a) c ---
# SHORT TERM BEHAVIOURMSD_ballistic
fig, ax = plt.subplots()

ax.plot(time_arr, MSD_linear, label=r"MSD linear: $4 \, D_t \, t$")
ax.plot(time_arr, MSD_ballistic, linestyle='dotted', label=r"MSD ballistic: $v_0^2 \, t^2$")
ax.plot(time_arr, MSD_numerical, label=r"MSD numerical")
ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim((1e-2,t_sim))
# ax.set_ylim((1e-2,25))
ax.legend(loc="best")
plt.show()
# plt.savefig(os.path.join(plots_path, "2a_c_MSD_ballistic.pdf"))
plt.close()


#%%
# LONG TERM BEHAVIOUR 
fname = "outputs/traj_positions_n500_tsim100_dt0.001_v05_Dt1_Dr10.txt"
Dt = 1
Dr = 10
v0 = 5
dt = 0.001
t_sim = 100

D_eff = Dt + v0**2 / (2*Dr)
msd_diffusive = lambda t, D_eff: 4 * D_eff * t
msd_diffusive = lambda t, D_eff: 4 * D_eff * t
MSD_diffusive = msd_diffusive(time_arr, D_eff) 

msd_simple_theory = lambda t, D_t: 4 * D_t * t
MSD_simple_theory = msd_diffusive(time_arr, Dt) 

start_time = time()
print("Started loading trajectory and MSD calculation...")
traj_sim = load_positions_txt(filename=fname)
MSD_numerical = msd_numerical(traj_sim)
print(f"Finished loading trajectory and MSD calculation with a time of {time() - start_time:.2f} s")

fig, ax = plt.subplots()

ax.plot(time_arr, MSD_simple_theory, label=r"MSD simple: $4 \, D_t \, t$")
ax.plot(time_arr, MSD_diffusive, label=r"MSD diffusive: $4 \, D_\text{eff} \, t$")
ax.plot(time_arr, MSD_numerical, label=r"MSD numerical")
ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
ax.set_xlim((0,100))
ax.legend(loc="best")
# plt.show()
plt.savefig(os.path.join(plots_path, "2a_c_MSD_diffusive.pdf"))

#%%
# --------- Task 2 b) --------- #
# --------- MSD weakly <--> highly active Brownian Particles --------- #

# v0 of [1, 3, 6, 10, 15, 20]

v0_arr = np.array([1, 3, 6, 10, 15, 20])      # in units of sigma/tau
Dt = 1;         # in units of sigma**2/tau_BD
Dr = 1          # in units of 1/tau_BD
D_eff_arr = Dt +  v0_arr**2 / (2*Dr)        # in units of sigma**2/tau_BD
dt = 1e-3
n_save = 10

root_fname = "outputs/task_2a_a/traj_positions_n500_tsim100_dt0.001_v01_Dt1_Dr1.txt"
traj_position = {}
MSDs_numerical = {}


start_time = time()
print("Start loading positions...")
for D_eff, v0 in zip(D_eff_arr, v0_arr):
    fname = root_fname.replace("_v01", f"_v0{v0}", 1)
    print(fname)
    traj_position[v0] = load_positions_txt(filename=fname)
    MSDs_numerical[v0] = msd_numerical(traj_position[v0])

print(f"Finished loading trajectory and MSD calculation with a time of {time() - start_time:.2f} s")


msd_diffusive = lambda t, D_eff: 4 * D_eff * t

dt_saved = dt * n_save
t = np.arange(len(next(iter(MSDs_numerical.values())))) * dt_saved


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
