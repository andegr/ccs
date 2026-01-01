#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import calculate_msd, orientation_autocorrelation
from SaveToFile import load_orientations_txt, load_positions_txt
from Plot import set_Plot_Font
from time import time
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")


#%%
traj_orientation = load_orientations_txt(filename="outputs/traj_orientations.txt")

start_time = time()
print("Starting autocorrelation calculation...")
acorr = orientation_autocorrelation(traj_orientation)
print(f"Finished autocorrelation calculation with a time of {time() - start_time:.2f} s")
time_arr = np.arange(start=0, stop=len(acorr))

#%%
# --- PLOTTING ---
fig, ax = plt.subplots()

# # Plot all EM autocorrelations
ax.plot(time_arr, acorr, label=r"C(t)")

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$C(t)$")
ax.set_xlim(left=0, right=10000)
ax.legend(loc="best")
plt.show()
# plt.savefig(os.path.join(plots_path, "acorr.pdf"))


#%%
# --------- Task 2 a) --------- #
# --------- MSD Calculation Validation --------- #

Dt = parameters.Dt

dt = parameters.dt
t_sim = parameters.t_sim        # in units of tau_BD

start_time = time()
print("Started loading trajectory and MSD calculation...")
traj_sim = load_positions_txt(filename="outputs/traj_positions_n500_dt1e-03.txt")
MSD_numerical = calculate_msd(traj_sim)
print(f"Finished loading trajectory and MSD calculation with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save  #
time_arr = np.arange(traj_sim.shape[2]) * dt_saved

n_steps_saved = parameters.n_steps_saved
MSD_theory = lambda t, Dt: 4 * Dt * t
MSD_theo = MSD_theory(time_arr, Dt)

#%%
# --- PLOTTING ---
fig, ax = plt.subplots()

# # Plot all EM autocorrelations
ax.plot(time_arr, MSD_numerical, label=r"MSD numerical")
ax.plot(time_arr, MSD_theo, label=r"MSD theory: $4 \, D_t \, t$")

ax.set_xlabel(r"$t$ [$\tau_\text{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
# ax.set_xlim((0,10000))
ax.legend(loc="best")
plt.show()


# %%
