#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parameters import MDSimulationParameters
from observable_calculations import calculate_average_msd
from SaveToFile import load_runs
from Plot import set_Plot_Font
import os
set_Plot_Font()
os.chdir(os.path.dirname(__file__))

parameters = MDSimulationParameters()
plots_path = os.path.join(os.path.dirname(__file__), "plots")


#%%
# --------- Task 2 b) --------- #
# --------- MSD weakly <--> highly active Brownian Particles --------- #

v0_arr = np.array([0, 4, 8, 11, 15, 17, 20])      # in units of sigma/tau

n_particles = 500
t_sim = 100
dt = 0.001
Dt = 1
Dr = 1
n_runs = 10 

D_eff_theo_arr = Dt +  v0_arr**2 / (2*Dr)        # in units of sigma**2/tau_BD


def linear_msd(t, D_eff, offset):
    return 4 * D_eff * t + offset



MSDs_numerical = {}
D_eff_dict = {}
offset_dict = {}
cov_dict = {}

for v0 in v0_arr:
    t, traj_list = load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs)
    mean_msd = calculate_average_msd(traj_list)

    MSDs_numerical[v0] = mean_msd

    # exclude first points ( only long term regime)
    t_min = 40
    popt, pcov = curve_fit(
        linear_msd,
        t[t>t_min],
        mean_msd[t>t_min],
    )                                                  

    D_eff, offset = popt                              

    D_eff_dict[v0] = D_eff                            
    offset_dict[v0] = offset                          
    cov_dict[v0] = pcov                             


msd_diffusive = lambda t, D_eff: 4 * D_eff * t

#%%
#------plotting fit----------------
fig, ax = plt.subplots()

for D_eff_theo, v0 in zip(D_eff_theo_arr, v0_arr):
    line_sim, = ax.plot(t, MSDs_numerical[v0], 
                        label=r"$D_\text{eff, fit}$"+ f"={D_eff_dict[v0]:.1f}," + r"$D_\text{eff, theo}$" + f"={D_eff_theo}, " \
                        + "$c$"+ f"={offset_dict[v0]:.0f}," + "$v_0$" + f"={v0}")
    c = line_sim.get_color()

    # theory (same color as sim, but no legend entry)
    MSD_diffusive = linear_msd(t, D_eff_dict[v0], offset_dict[v0]) 
    ax.plot(t, MSD_diffusive, linestyle="--", color=c, label="_nolegend_")

ax.set_xlabel(r"$t$ [$\tau_\mathrm{BD}$]")
ax.set_ylabel(r"$MSD(t)$")
ax.set_xlim((0, 100))
# ax.set_ylim(5e-2, 1.1)   # Beispiel
ax.legend()
plt.savefig(os.path.join(plots_path, "MSD_activity_fit.pdf"))


# %%
#-----plotting Deff over Pe_eff--------------------

peclet_array = v0_arr**2 / (2 * Dr * Dt)

fig, ax = plt.subplots()


ax.plot( peclet_array, Dt*(1+peclet_array), linestyle="--", label="Theory")

D_eff_vals = np.array([D_eff_dict[v0] for v0 in v0_arr]) 
ax.scatter(peclet_array, D_eff_vals, label="Fit")           


ax.set_xlabel(r"$Pe_{eff}$ UNIT")
ax.set_ylabel(r"$D_{eff}$ UNIT")
# ax.set_xlim((0, 100))
# ax.set_ylim(5e-2, 1.1)
ax.legend()
plt.savefig(os.path.join(plots_path, "deff_over_peclet.pdf"))
# %%
