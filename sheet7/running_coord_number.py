import numpy as np
import matplotlib.pyplot as plt

from Plot import apply_style
from SaveToFile import load_hist
import os

apply_style()

filename = f"sheet7/hist_0.5_1.txt"
hist, dr = load_hist(filename)

rho = 0.5
r = np.arange(len(hist)) * dr

# --- Compute n(r_c^i) using the trapezoid rule ---
integrand = hist * r**2
n_rc = 4 * np.pi * rho * np.cumsum((integrand[:-1] + integrand[1:]) * 0.5 * dr)

# remove first support point
r_c = r[1:]

plt.plot( r_c, r_c**3, label=r"$r_c^3$")
plt.plot(r_c, n_rc, label=r"$n(r_c)$")
plt.xlabel(r"$r_c$")
plt.ylabel(r"$n(r_c)$")
plt.legend()
plt.yscale("log")
plt.xlim(0,4.9)
plt.tight_layout()
plt.show()
