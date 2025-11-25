#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_positions_txt
import Plot
from parameters import L, kB, T, n_particles_task1
import os
import time
import logging
from IntegrationSchemes import pbc_distance, U_LJ

Plot.apply_style()

plots_path = os.path.join(os.path.dirname(__file__), "plots")
#%%
# --------------- Task 1 a) and b) ---------------- # 

x1 = L/8
x2 = L/11

x1 = L / 8
x2_start = L / 11
dx = -0.001
total_displacement = -L / 2

# number of steps
n_steps = int(abs(total_displacement / dx))  # around 5000 steps

# displacement values
displacements = dx * np.arange(n_steps + 1)

# raw positions
x2_positions = x2_start + displacements
print(x2_positions)

print(x2_start)

# --------------- Task 1 c) (i) ---------------- # 

min_distance_12 = np.zeros(shape=x2_positions.shape)

for i in range(len(x2_positions)):
    min_distance_12[i] = pbc_distance(xi=x1, xj=x2_positions[i], xlo=0, xhi=L)

abs_min_distance_12 = np.abs(min_distance_12)

# --- PLOTTING (i) ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x2_positions, abs_min_distance_12, label=r"$r_{21} = |x_2 - x_1 |$", color="black")
# ax.vlines(x=-L/2 + x1, ymin=0, ymax=5.5, label=r'-L/2 + $x_1$ = -L/2 + L/8 = -3L/8 =' + f'{-L/2 + x1:.2f}', 
        #   linestyles="dotted", colors='green')
ax.vlines(x=x1, ymin=0, ymax=5.5, label=r'$x_1=L/8 =$' + f'{x1:.2f}', colors='green')
# ax.vlines(x=-L/2, ymin=0, ymax=5.5, label=r'PBC $L/2 =$' + f'{-L/2:.1f}',
#           linestyles="dotted", colors='red')
ax.hlines(y=1.25, xmin=x2_positions[-1], xmax=x2_start, label=r'$\Delta x_2=L/2$', colors = 'blue')
ax.vlines(x=[x2_positions[-1], x2_positions[0]], ymin=0, ymax=5.5, 
          colors='blue', linestyles="dotted")

ax.set_xlabel(r"$x_2$")
ax.set_ylabel(r"$r = |x_2 - x_1 |$")
ax.set_xlim(left=-4.75, right=2)
# bbox_to_anchor places the legend relative to the axes. (1.05, 1) is just outside the top-right.
# ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
ax.legend(loc='upper center')
plt.tight_layout() # Adjust layout to ensure nothing is clipped
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_path, "Part_1_c_i.pdf"))
# plt.show()



# --------------- Task 1 c) (ii) ---------------- # 

# preparing positions of shape = numb_particles, dimensions

positions = np.zeros((2, 1))
positions[0, 0] = x1
positions[1, 0] = x2_positions[i]


U_L_list = []

for x2 in x2_positions:
    positions = np.zeros((2, 1))
    positions[0, 0] = x1
    positions[1, 0] = x2
    U_L_list.append(U_LJ(positions))

# plt.plot(abs_min_distance_12, U_L_list)
# plt.show()


# --- PLOTTING (iii) ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(abs_min_distance_12, U_L_list, label=r"$U_\text{LJ}(r_{21})$", color="black")
ax.set_xlabel(r"r")
ax.set_ylabel(r"U_LJ(r)")
ax.legend(loc='upper center')
ax.set_xlim(0, 1.5)
plt.tight_layout() # Adjust layout to ensure nothing is clipped
ax.grid(True, alpha=0.3)
# plt.savefig(os.path.join(os.path.dirname(__file__), "plots/Part_1_d_iii_C0.1.pdf"))
plt.show()
