#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import msd_numerical, msd_theory, orientation_autocorrelation_averaged
from SaveToFile import load_orientations_txt, load_positions_txt
from Plot import set_Plot_Font, apply_style
from time import time
import os
set_Plot_Font()
apply_style()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")

#%%
# --------- Task 1) --------- #
# --------- Autocorrelation Validation --------- #
Dr_list = [1, 2, 5, 10, 20]
dt = 1e-3
t_sim = 200
block_size = 100 # block size of a_corr averaging

root_fname = "outputs/traj_orientations_n500_tsim200_teq20_dt0.001_v00_Dt1_Dr1_run0.txt"
traj_orientation = {}
acorr = {}

acorr_exp_decay = lambda t, Dr: np.exp(-t*Dr)

start_time = time()
print("Starting loading + autocorrelation calculation...")

for Dr in Dr_list:
    fname = root_fname.replace("_Dr1", f"_Dr{Dr}", 1)
    time_arr, traj_orientation[Dr] = load_orientations_txt(fname, dt, t_sim)
    dt_saved = time_arr[1] - time_arr[0]
    print(len(time_arr), len(traj_orientation[Dr]))
    t_corr, acorr[Dr] = orientation_autocorrelation_averaged(traj_orientation[Dr], dt_saved, blocks=block_size)

print(f"Finished with a time of {time() - start_time:.2f} s")

#%%
# # --- PLOTTING ---
fig, ax = plt.subplots()

# one legend entry for theory (black dashed)
ax.plot([], [], "k--", label="theory")

for Dr in Dr_list:
    line_sim, = ax.plot(t_corr, acorr[Dr], label=rf"$D_r={Dr}$")
    c = line_sim.get_color()

    # theory (same color as sim, but no legend entry)
    ax.plot(t_corr, acorr_exp_decay(t_corr, Dr), linestyle="--", color=c, label="_nolegend_")

ax.set_xlabel(r"$t$ / $\tau_\mathrm{BD}$")
ax.set_ylabel(r"$C_{\hat{\mathbf{n}}}(t)$")
ax.set_yscale("log")
ax.set_xlim((0, 1.2))
ax.set_ylim(5e-3, 1.1)   # Beispiel
# ax.legend()
ax.legend(
    loc="center left",
    bbox_to_anchor=(.92, 1.05),
    frameon=True
)
# plt.tight_layout(rect=[0, 0, 0.8, 1])
# plt.show()
plt.savefig(os.path.join(plots_path, "1_acorr_orientation.png"))

#%%
# --------- Task 2a b) --------- #
# --------- Autocorrelation Function Validation --------- #
fname = "outputs/traj_orientations_n500_tsim200_dt0.001_v01_Dt1_Dr10.txt"
# v0 = 1
# Dt = 1
# Dr = 10
# dt = 0.001
# t_sim = 100        # in units of tau_BD

# Dr = 10

# start_time = time()
# print("Starting loading + autocorrelation calculation...")
# traj_orientation = load_orientations_txt(filename=fname)
# acorr = orientation_autocorrelation_averaged(traj_orientation)
# print(f"Finished with a time of {time() - start_time:.2f} s")

# dt_saved = parameters.dt * parameters.n_save
# t = np.arange(len(acorr)) * dt_saved


# fig, ax = plt.subplots()

# ax.plot(t, acorr, label=rf"$D_r={Dr}\,\mathrm{{BD}}$", color="blue")

# ax.plot(t, acorr_exp_decay(t, Dr), linestyle="--", color="blue", label="theory")
# ax.set_xlabel(r"$t$ [$\tau_\mathrm{BD}$]")
# ax.set_ylabel(r"$C(t)$")
# ax.set_yscale("log")
# ax.set_xlim((0, 0.75))
# # ax.set_ylim(5e-4, 1.1)   # Beispiel
# ax.legend()
# plt.show()
# plt.savefig(os.path.join(plots_path, "2a_b_acorr_orientation.pdf"))
# %%
