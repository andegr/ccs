#%%
import numpy as np
import matplotlib.pyplot as plt
from parameters import MDSimulationParameters
from observable_calculations import one_particle_density, average_one_particle_density
from SaveToFile import load_runs
from Plot import set_Plot_Font, apply_style
from time import time
import os
# set_Plot_Font()
apply_style()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")

#%%
# ------ Loading ------- #


L = 30
t_sim = 100
t_eq = 25
dt = 0.0001
n_save = 1000
n_particles = 250
Dt = 1
n_runs = 1
eta = 0.3

r_disc = 15
epsilon_disc = 10

v0_arr = np.array([0, 10, 20, 35]) #np.array([50, 35, 20, 10, 0]) # 0, 10, 20,
Dr_arr = np.array([1])
Dt_arr = np.array([1])

rho_r_dict = {}

for v0 in v0_arr:
    for Dr in Dr_arr:
        for Dt in Dt_arr:
            t, traj_list = load_runs(n_particles,
                                    t_sim,
                                    t_eq,
                                    dt,
                                    L,
                                    v0,
                                    Dt,
                                    Dr,
                                    n_runs,
                                    walls = False,
                                    pairwise = True,
                                    eta = eta,
                                    disc = True,
                                    r_disc = r_disc,
                                    epsilon_disc = epsilon_disc,
                                    )

            bin_centers, rho_r = average_one_particle_density(
            traj_list,
            n_bins=300,
            x_range= (0, 16),    # only allow bins in this range and fix therefore for each run
            direction = "radial",
            L = L,
            )

            rho_r_dict[v0, Dt, Dr] = rho_r

#%%

fig, ax = plt.subplots()

for v0 in v0_arr:
    ax.plot(bin_centers, rho_r_dict[v0, 1, 1], linestyle="-", label=rf"$v_0$ = {v0}")
# ax.plot(bin_centers, box_density(bin_centers,L), linestyle="--", label=r"theory")

# ax.axhline(1.0 / L, linestyle="--", linewidth=2, label=r"$\rho_0$")


ax.set_xlabel(r"$r$ / $\sigma$ ")
ax.set_ylabel(r"$\rho(r)$")
ax.set_xlim((0,r_disc +1))
# ax.set_ylim((0,0.1))
ax.grid(which='both', axis='both')
ax.legend(loc="best")
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(plots_path, "disc_activity_dist.pdf"))



# %%


import matplotlib.pyplot as plt

# x-limits to keep (cut out the middle)
x1 = (0, 2)
x2 = (9, 15)

# NOTE: for an x-axis break you need 1 row, 2 columns (side-by-side), not (2,1)
fig, (ax, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(8, 5),
    gridspec_kw={"width_ratios": [1, 4]}  # left ~1/3, right ~2/3
)

linestyles = ["-", "--", "-.", ":"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]  # easy to distinguish

for i, v0 in enumerate(v0_arr):
    y = rho_r_dict[v0, 1, 1]
    ls = linestyles[i % len(linestyles)]
    c  = colors[i % len(colors)]
    ax.plot(bin_centers, y, linestyle=ls, color=c, linewidth=2.0, label=rf"$v_0$ = {v0}")
    ax2.plot(bin_centers, y, linestyle=ls, color=c, linewidth=2.0, label=rf"$v_0$ = {v0}")

ax.set_xlim(x1)
ax2.set_xlim(x2)

# ax.set_xlabel(r"$r$ / $\sigma$")
ax2.set_xlabel(r"$r$ / $\sigma$")
ax.set_ylabel(r"$\rho(r)$")

ax.grid(which='both', axis='both')
ax2.grid(which='both', axis='both')

# ax.spines["right"].set_visible(False)
# ax2.spines["left"].set_visible(False)
ax.tick_params(right=False)
ax2.tick_params(left=False)

ax2.legend(loc=2)

# draw the // break marks ON THE X-AXIS boundary between the two panels
# d = 0.02
# kwargs = dict(color="k", clip_on=False, linewidth=1.5)

# make BOTH sides use the same slope and symmetric coords
# ax.plot((1-d, 1+d), (-d, +d), transform=ax.transAxes, **kwargs)
# ax.plot((1-d, 1+d), (1-d, 1+d), transform=ax.transAxes, **kwargs)

# ax2.plot((-d, +d), (-d, +d), transform=ax2.transAxes, **kwargs)
# ax2.plot((-d, +d), (1-d, 1+d), transform=ax2.transAxes, **kwargs)
# ax.set_aspect("auto")
# ax2.set_aspect("auto")

d = 0.01
lw = 1.


plt.subplots_adjust(wspace=0.1)   # smaller gap (try 0.0–0.03)

# draw in *figure* coordinates so both slashes have identical tilt
p1 = ax.get_position()
p2 = ax2.get_position()

fig.lines.extend([
    plt.Line2D([p1.x1-d, p1.x1+d], [p1.y1-d, p1.y1+d], transform=fig.transFigure, color="k", lw=lw),
    plt.Line2D([p1.x1-d, p1.x1+d], [p1.y0-d, p1.y0+d], transform=fig.transFigure, color="k", lw=lw),
    plt.Line2D([p2.x0-d, p2.x0+d], [p2.y1-d, p2.y1+d], transform=fig.transFigure, color="k", lw=lw),
    plt.Line2D([p2.x0-d, p2.x0+d], [p2.y0-d, p2.y0+d], transform=fig.transFigure, color="k", lw=lw),
])
# plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plots_path, "disc_activity_dist.png"), dpi=150)


# %%
