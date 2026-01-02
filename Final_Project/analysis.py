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
# --------- Task 1) --------- #
# --------- Autocorrelation Validation --------- #
Dr_list = [1, 2, 5, 10, 25]

root_fname = "outputs/traj_orientations_n500_tsim100_dt0.001_v00_Dt1_Dr1.txt"
traj_orientation = {}
acorr = {}

acorr_exp_decay = lambda t, Dr: np.exp(-t*Dr)

start_time = time()
print("Starting loading + autocorrelation calculation...")

for Dr in Dr_list:
    fname = root_fname.replace("_Dr1.txt", f"_Dr{Dr}.txt", 1)
    traj_orientation[Dr] = load_orientations_txt(filename=fname)
    acorr[Dr] = orientation_autocorrelation(traj_orientation[Dr])

print(f"Finished with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save
t = np.arange(len(next(iter(acorr.values())))) * dt_saved


#%%
# # --- PLOTTING ---
fig, ax = plt.subplots()

# one legend entry for theory (black dashed)
ax.plot([], [], "k--", label="theory")
clipping_value = 1e-3  # oder kleiner/größer je nach Noise-Level


for Dr in Dr_list:
    # simulation
    # acorr_clipped = np.clip(acorr[Dr], clipping_value, None)    line_sim, = ax.plot(t, acorr[Dr], label=rf"$D_r={Dr}$")
    # line_sim, = ax.plot(t, acorr_clipped, label=rf"$D_r={Dr}$")
    line_sim, = ax.plot(t, acorr[Dr], label=rf"$D_r={Dr}$")
    c = line_sim.get_color()

    # theory (same color as sim, but no legend entry)
    ax.plot(t, acorr_exp_decay(t, Dr), linestyle="--", color=c, label="_nolegend_")

ax.set_xlabel(r"$t$ [$\tau_\mathrm{BD}$]")
ax.set_ylabel(r"$C(t)$")
ax.set_yscale("log")
ax.set_xlim((0, 5.5))
ax.set_ylim(5e-2, 1.1)   # Beispiel
ax.legend()
plt.savefig(os.path.join(plots_path, "1_acorr_orientation.pdf"))

#%%
# --------- Task 2a b) --------- #
# --------- Autocorrelation Function Validation --------- #
fname = "outputs/traj_orientations_n500_tsim100_dt0.001_v01_Dt1_Dr10.txt"
v0 = 1
Dt = 1
Dr = 10
dt = 0.001
t_sim = 100        # in units of tau_BD

start_time = time()
print("Starting loading + autocorrelation calculation...")
traj_orientation = load_orientations_txt(filename=fname)
acorr = orientation_autocorrelation(traj_orientation)
print(f"Finished with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save
t = np.arange(len(acorr)) * dt_saved


fig, ax = plt.subplots()

ax.plot(t, acorr, label=rf"$D_r={Dr}$", color="blue")
ax.plot(t, acorr_exp_decay(t, Dr), linestyle="--", color="blue", label="theory")
ax.set_xlabel(r"$t$ [$\tau_\mathrm{BD}$]")
ax.set_ylabel(r"$C(t)$")
ax.set_yscale("log")
ax.set_xlim((0, 0.75))
ax.set_ylim(5e-2, 1.1)   # Beispiel
ax.legend()
plt.savefig(os.path.join(plots_path, "2a_b_acorr_orientation.pdf"))


#%%
# --------- Task 2a a) --------- #
# --------- MSD Calculation Validation --------- #

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


fname = "outputs/traj_positions_n500_tsim100_dt0.001_v02_Dt1_Dr1.txt"
Dt = 1
Dr = 1
v0 = 2
dt = 0.001
t_sim = 100

start_time = time()
print("Started loading trajectory and MSD calculation...")
traj_sim = load_positions_txt(filename=fname)
MSD_numerical = msd_numerical(traj_sim)
print(f"Finished loading trajectory and MSD calculation with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save  #
time_arr = np.arange(traj_sim.shape[2]) * dt_saved

msd_ballistic = lambda t, Dt, v0: 4 * Dt * t + v0**2 * t**2
MSD_ballistic = msd_ballistic(time_arr, Dt, v0)

# --- PLOTTING 2c) ---

# SHORT TERM BEHAVIOUR
fig, ax = plt.subplots()

ax.plot(time_arr, MSD_ballistic, label=r"MSD ballistic: $4 \, D_t \, t + v_0^2 \, t^2$")
ax.plot(time_arr, MSD_numerical, label=r"MSD numerical")
ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
ax.set_xlim((0,2))
ax.set_ylim((0,25))
ax.legend(loc="best")
# plt.show()
plt.savefig(os.path.join(plots_path, "2a_c_MSD_ballistic.pdf"))
plt.close()


# LONG TERM BEHAVIOUR 

fname = "outputs/traj_positions_n500_tsim100_dt0.001_v05_Dt1_Dr10.txt"
Dt = 1
Dr = 10
v0 = 5
dt = 0.001
t_sim = 100

D_eff = Dt + v0**2 / (2*Dr)
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
# --------- MSD Calculation Validation --------- #