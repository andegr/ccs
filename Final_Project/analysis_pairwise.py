#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from helpers import fmt_float
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

n_particles = 250
sigma = 1
area_fracs = [0.3, 0.3 + 1/30, 0.3 + 2/30, 0.4, 0.4+1/30, 0.4+2/30, 0.5]
L_arr = np.sqrt( n_particles * np.pi * sigma**2 / (4*area_fracs))

v0 = 0
t_sim = 100
t_eq = 100
dt = 0.0001
n_save = 100
Dt = 1
Dr = 1
n_runs = 1


root_fname = "outputs/traj_positions_.txt"
traj_positions = {}

start_time = time()
print("Starting loading + autocorrelation calculation...")

for eta, L in zip(area_fracs, L_arr):
    fname = root_fname.replace("_eta", f"_eta{fmt_float(eta, max_decimals=2)}.txt", 1)
    fname = root_fname.replace("_L", f"_eta{fmt_float(L, max_decimals=1)}.txt", 1)
    for run in range(n_runs):
        fname = root_fname.replace("_run0.txt", f"_run{run}.txt", 1)
        traj_positions[eta] = load_positions_txt(filename=fname)

print(f"Finished with a time of {time() - start_time:.2f} s")

dt_saved = parameters.dt * parameters.n_save


#%%
# # --- PLOTTING ---
fig, ax = plt.subplots()

# one legend entry for theory (black dashed)
# ax.plot([], [], "k--", label="theory")

for eta in area_fracs:
    pass
    # simulation
    # acorr_clipped = np.clip(acorr[Dr], clipping_value, None)    line_sim, = ax.plot(t, acorr[Dr], label=rf"$D_r={Dr}$")
    # line_sim, = ax.plot(t, acorr_clipped, label=rf"$D_r={Dr}$")
    # line_sim, = ax.plot(t, acorr[Dr], label=rf"$D_r={Dr}$")
    # c = line_sim.get_color()

    # theory (same color as sim, but no legend entry)
    # ax.plot(t, acorr_exp_decay(t, Dr), linestyle="--", color=c, label="_nolegend_")

ax.set_xlabel(r"$t$ [$\tau_\mathrm{BD}$]")
ax.set_ylabel(r"$C(t)$")
# ax.set_yscale("log")
# ax.set_xlim((0, 5.5))
# ax.set_ylim(5e-4, 1.1)   # Beispiel
ax.legend()
plt.show()