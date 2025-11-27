#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_positions_txt
import Plot
from parameters import xlo, xhi, eps, kB, T, sigma
from IntegrationSchemes import pbc_distance_array
import os
import time
import logging

Plot.apply_style()
#%%

# --------------- Task 2 b) ---------------- # 
plots_path = os.path.join(os.path.dirname(__file__), "plots")

print(os.path.join(plots_path, "Task_2b.pdf"))

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
hist_P_of_r, bins = np.histogram(abs_r_12_distances, bins=200, density=True)
bin_centers = 0.5*(bins[:-1] + bins[1:])


# %%
# --------------- Task 2 c) ---------------- # 

# Calculating C

min_U_raw = np.min(- kB * T * np.log(hist_P_of_r))
C = -min_U_raw - eps

U_PMF = -kB*T * np.log(hist_P_of_r) + C


# --------------- Plotting Task 2 b), c) ---------------- # 
fig, ax = plt.subplots(figsize=(12,6))

# # Plot all EM autocorrelations
ax.plot(bin_centers, hist_P_of_r, label=r"$P(r)$", color='#0F31DA')
ax.plot(bin_centers, U_PMF, label=r'$U_\text{PMF}(r)$', linestyle="solid", color='#F98500')
ax.set_xlabel(r"$r / \sigma$")
ax.set_ylabel(r"$P(r)$, $U_\text{PMF}(r)$")
ax.set_ylim(-1.25, 2)
ax.legend(loc="best")
plt.savefig(os.path.join(plots_path, "Task_2bc.pdf"))
plt.show()


# %%
# --------------- Task 2 d) ---------------- # 

def U_LJ_func(r):
    U_LJ = 4.0 * eps * ((sigma / r)**12 - (sigma / r)**6)
    return U_LJ

U_LJ = U_LJ_func(abs_r_12_distances)


# 1. Digitize: Determine which bin each distance (and its corresponding energy) falls into.
# The 'digitize' function returns the index of the bin each element belongs to.
# x = np.array([0.2, 6.4, 3.0, 1.6])
# bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
# inds = np.digitize(x, bins)
# output:
# array([1, 4, 3, 2])
bin_indices = np.digitize(abs_r_12_distances, bins)     # uses same bins as before for P(r)
# --> bin_indices is array like r_12 and in the same order and and has values of the 
# index with regard to the bins. Example: r_12[5] = 1.8, und bins = [0, 1, 2, ...] dann wäre
# bin_indices[5] = 2 , 

# 2. Group and Average: Use np.bincount to sum the energies and count the occurrences
# within each bin index.

# a) Sum the energies (numerator) in each bin
sum_U_per_bin = np.bincount(bin_indices, weights=U_LJ)  
# weights (multiplies) every bin_idx with accourding U_(r) weight and then counts up all 
# entries of bin_indices which have the same index and summs the weighted entries, i.e.
# summs up all 

# b) Count the number of entries (denominator) in each bin
counts_per_bin = np.bincount(bin_indices)

# 3. Calculate the Average (Handle zero counts to avoid division by zero)
# The resulting array will have length len(bins) + 1, where index 0 is for values outside
# the first bin, and the last index is for values outside the last bin.

# Create a container for the final result
u_r_avg = np.zeros_like(sum_U_per_bin, dtype=float)

# Only calculate the average where the bin count is non-zero
# valid_indices = counts_per_bin > 0        # not needed
u_r_avg[bin_indices] = sum_U_per_bin[bin_indices] / counts_per_bin[bin_indices]


bin_centers = 0.5 * (bins[:-1] + bins[1:])  # this actually skips 1st and last bin so we need then...
u_r_plot = u_r_avg[1:len(bins)]

# --------------- Plotting Task 2 d) ---------------- # 
fig, ax = plt.subplots(figsize=(12,6))

# # Plot all EM autocorrelations
ax.plot(bin_centers, u_r_plot, label=r"$u(r)=\langle U_\text{LJ}(r) \rangle$", color="#0F31DA")
ax.plot(bin_centers, U_PMF, label=r"$U_\text{PMF}(r)$", color="#F98500",  linestyle='dashed')
ax.set_xlabel(r"$r / \sigma$")
ax.set_ylabel(r"$P(r)$, $U_\text{PMF}(r)$")
# ax.set_ylim(-1.25, 2)
ax.legend(loc="best")
plt.savefig(os.path.join(plots_path, "Task_2d.pdf"))
plt.show()


# %%
import numpy as np

x = np.array([0, 2, 0, 2, 2, 3])
w = np.array(x) -1  # weights

a = np.bincount(x,  weights=w)
print(a)

b = np.bincount(x)
print(b)
# %%
