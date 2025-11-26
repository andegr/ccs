#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_positions_txt
import Plot
from parameters import xlo, xhi
from IntegrationSchemes import pbc_distance_array
import os
import time
import logging

Plot.apply_style()
#%%
plots_path = os.path.join(os.path.dirname(__file__), "plots")

# changing path to sheet 5
os.chdir(os.path.dirname(__file__))

start_time = time.time()
positions = load_positions_txt("trajectory_1000tauProdRun.txt")
print(f"Finished loading trajectory with total time of {time.time() - start_time:.2f} s")

x1_time_series = positions[0, 0, :]   # (T,)
x2_time_series = positions[1, 0, :]   # (T,)

r_12_distances = pbc_distance_array(x1_time_series, x2_time_series, xlo, xhi)
abs_r_12_distances = np.abs(r_12_distances)
# Histogram
hist, bins = np.histogram(abs_r_12_distances, bins=200, density=True) # just taking the first particle with [0]
bin_centers = 0.5*(bins[:-1] + bins[1:])

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(12,6))

# # Plot all EM autocorrelations
ax.plot(bin_centers, hist, label=r"Simulation Data")

ax.set_xlabel(r"$r_{12}$")
ax.set_ylabel(r"$P(r_{12})$")
# ax.set_xlim(left=-0.1)
ax.legend(loc="best")
plt.savefig(os.path.join(plots_path, "Task_2b.pdf"))
plt.show()
# %%
