import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Parameters
Dt = 1.0
dt = 1.0
sigma = np.sqrt(2 * Dt * dt)

# Domain exactly up to |x|,|y| = 1
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# 2D Gaussian PDF
P = (1 / (4 * np.pi * Dt * dt)) * np.exp(-(X**2 + Y**2) / (4 * Dt * dt))

# PDF value at r = 1 (used as fade-to-white reference)
P_edge = (1 / (4 * np.pi * Dt * dt)) * np.exp(-1 / (4 * Dt * dt))

# Colormap normalization:
# values <= P_edge → white
norm = Normalize(vmin=P_edge, vmax=P.max())

plt.figure(figsize=(5, 5))
plt.imshow(
    P,
    extent=[-1, 1, -1, 1],
    origin="lower",
    cmap="Blues",
    norm=norm,
    alpha=0.8
)

plt.xlabel(r"$\Delta x$", fontsize=18)
plt.ylabel(r"$\Delta y$", fontsize=18)
plt.xticks([-1.0, -0.5, 0, 0.5, 1.0], fontsize=14)
plt.yticks([-1.0, -0.5, 0, 0.5, 1.0], fontsize=14)

plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from Plot import set_Plot_Font, apply_style

apply_style()

def get_potentials(epsilon=1.0, sigma=1.0):
    # Determine the cutoff for WCA (the minimum of LJ)
    r_min = sigma * 2**(1/6)
    
    # Distance range (avoiding zero to prevent division by zero)
    r = np.linspace(0.95 * sigma, 2.5 * sigma, 500)
    
    # Calculate LJ potential
    # Formula: 4 * epsilon * ((sigma/r)^12 - (sigma/r)^6)
    inv_r_six = (sigma / r)**6
    v_lj = 4 * epsilon * (inv_r_six**2 - inv_r_six)
    
    # Calculate WCA potential
    # It is the LJ potential shifted up by epsilon and truncated at r_min
    v_wca = np.where(r <= r_min, v_lj + epsilon, 0)
    
    return r, v_lj, v_wca, r_min

def plot_comparison():
    epsilon = 1.0
    sigma = 1.0
    r, v_lj, v_wca, r_min = get_potentials(epsilon, sigma)

    plt.figure(figsize=(6, 5))

    # Plot LJ potential as dotted
    plt.plot(r, v_lj, label='Lennard-Jones', color='black', linestyle=':', linewidth=2)
    
    # Plot WCA potential
    plt.plot(r, v_wca, label='WCA (Repulsive)', color='#d62728', linestyle='-', linewidth=2)

    # Visual aids
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(r_min, color='blue', linestyle='--', alpha=0.5, label=r'Cutoff ($2^{1/6}\sigma$)')
    
    # Styling
    # plt.title('Comparison of Lennard-Jones and WCA Potentials', fontsize=12)
    plt.xlabel(r'$r \, / \, \sigma$')
    plt.ylabel(r'$U(r) \, / \, \epsilon$')
    plt.ylim(-1.2, 3.0)
    plt.xlim(0.9, 2.2)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Tight layout as requested
    plt.tight_layout()
    plt.show()


plot_comparison()
# %%
