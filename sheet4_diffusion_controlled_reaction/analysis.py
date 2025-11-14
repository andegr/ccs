import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from SaveToFile import load_trajectory, load_timesteps_and_observable
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import x_R, n_particles

positions = load_trajectory("trajectory.txt")

# Histogram
hist, bins = np.histogram(positions[0], bins=100, density=True)
hist *= n_particles
bin_centers = 0.5*(bins[:-1] + bins[1:])

# Analytical function ----------------------------------------------------
def one_body_density(x, rho0):
    if np.isscalar(x):
        if x < 0 or x > x_R:
            return 0
        return rho0 - rho0/x_R * x
    else:
        out = np.zeros_like(x)
        mask = (0 <= x) & (x <= x_R)
        out[mask] = rho0 - rho0/x_R * x[mask]
        return out

# Compute analytical curve using your original rho0 ----------------------
rho0_original = 2 * n_particles / x_R
analytical = one_body_density(bin_centers, rho0_original)

# Fit rho0 ---------------------------------------------------------------
popt, pcov = curve_fit(
    lambda x, rho0: one_body_density(x, rho0),
    bin_centers,
    hist,
    p0=[rho0_original]
)
rho0_fit = popt[0]
fitted_curve = one_body_density(bin_centers, rho0_fit)

# Residuals --------------------------------------------------------------
residuals = hist - fitted_curve

# Plot -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(7, 8),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    sharex=True
)

# Main plot
ax1.plot(bin_centers, hist, "g", label=r"Simulation $\Delta t = 0.001\,\tau_{BD}$")
ax1.plot(bin_centers, analytical, "r--", label=fr"Analytical ($\rho_0 = {rho0_original:.1f}$)")
ax1.plot(bin_centers, fitted_curve, "b", label=fr"Fit ($\rho_0 = {rho0_fit:.1f}$)")
ax1.set_ylabel(r"$\rho$")
ax1.legend(loc="upper right")
ax1.tick_params(labelbottom=False)

# Residual plot
ax2.axhline(0, color="black", linewidth=1)
ax2.plot(bin_centers, residuals, "r--", label="Residuals (data − fit)")
ax2.plot(bin_centers, analytical - fitted_curve, "b", label="Residuals (analytical − fit)")
ax2.set_xlabel(r"$x/\sigma$")
ax2.set_ylabel("Residuals")
ax2.legend(loc="upper right")

plt.show()

fig.savefig(os.path.join("plots", "stationary_distribution_with_fit.pdf"))
