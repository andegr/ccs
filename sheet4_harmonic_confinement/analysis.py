#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_timesteps_and_observable
from Plot import set_Plot_Font
set_Plot_Font()
import os
os.chdir(os.path.dirname(__file__))

from parameters import K_H, kB, T

timesteps_acorr, acorr = load_timesteps_and_observable("acorr.txt")
timesteps_acorr, acorr_OU = load_timesteps_and_observable("acorr_OU.txt")

plt.figure("1d)")

# Plot stationary distribution
plt.plot(timesteps_acorr[:200], acorr[:200], "g", label=r'$C_{xx, EM}$ for $\Delta t = 0.1\, \tau_{BD}, K_H=10$')
plt.plot(timesteps_acorr[:200], acorr_OU[:200], "b", label=r'$C_{xx, OU} $ for $\Delta t = 0.1\, \tau_{BD}, K_H=10$')
plt.legend()
plt.savefig(os.path.join("plots", "1d)_stat_distr_dt0.01_nsave1e5_KH10_tsim5e6_.pdf"))
# plt.show()

