import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory
import os
os.chdir(os.path.dirname(__file__))

positions = load_trajectory("trajectory.txt")

print(positions.dtype)
print(positions)

plt.hist(positions[0], density=True, bins=300, label="Simulation")
plt.xlabel(r"$x/\sigma$")
plt.ylabel(r"$\rho$")
plt.legend()
plt.show()