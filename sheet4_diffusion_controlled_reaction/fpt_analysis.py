import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from SaveToFile import load_observable
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import x_R, n_particles

fpts = load_observable("fpt.txt")

# Histogram
hist, bins = np.histogram(fpts, bins=100, density=True)
bin_centers = 0.5*(bins[:-1] + bins[1:])


plt.plot(bin_centers, hist, "g", label=r"Simulation $\Delta t = 0.001\,\tau_{BD}$")
plt.ylabel(r"p(t)")
plt.xlabel(r"$t\,/\,\tau_{BD}$")
plt.legend()
plt.show()
plt.savefig(os.path.join("plots", "fpt.pdf"))

fpt_mean = np.mean(fpts)
fpt_vaiance = np.std(fpts, ddof=-1)
k_mfpt = 1/ fpt_mean

print(f"mean: {fpt_mean:.2f}")
print(f"variance: {fpt_vaiance:.2f}")

#--------part ii)---------------------

# Compute the cumulative distribution F(t) from the histogram
cdf = np.cumsum(hist * np.diff(bins))  # F(t) = integral of p(t)
survival_prob = 1 - cdf             # S(t) = 1 - F(t)

# Plot survival probability
plt.figure()
plt.plot(bin_centers, survival_prob, "b", label="Survival probability S(t)")
plt.xlabel(r"$t\,/\,\tau_{BD}$")
plt.ylabel(r"$S(t)$")
plt.legend()
plt.show()
plt.savefig(os.path.join("plots", "survival_probability.pdf"))


#----------------- part iii) -----------------
# Compute mean FPT from survival probability
# Integral approximated numerically
dt = np.diff(bins)[0]         
mean_fpt_from_S = np.sum(survival_prob * dt)

# Compute k_FPT
k_fpt = 1 / mean_fpt_from_S

print(f"Mean FPT from survival probability: {mean_fpt_from_S:.2f}")
print(f"k_FPT = 1 / ⟨t_FPT⟩: {k_fpt:.4f}")

