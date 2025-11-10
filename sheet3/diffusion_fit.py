import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory
import os
os.chdir(os.path.dirname(__file__))

from numba import njit, prange

positions = load_trajectory("trajectory.txt")


@njit(parallel=True)
def create_one_body_distribution(positions, bins=40):

    n_analysis = positions.shape[2]
    n_particles = positions.shape[0]

    r_max = np.max(np.sqrt( positions[:,0,-1]**2 + positions[:,1,-1]**2))
    dr = r_max / (bins - 10)
    r_bins = np.arange(dr, dr*bins + dr, dr)

    one_body_dist = np.zeros((bins, n_analysis))

    for i in prange(n_analysis):
        r = np.sqrt( positions[:,0,i]**2 + positions[:,1,i]**2)

        for j in range(n_particles):
            idx = int(r[j] / dr)
            one_body_dist[idx,i] += 1
    # normalize it:
        one_body_dist[:,i] = one_body_dist[:,i] / (n_particles * 2*np.pi * r_bins * dr)
    

    return one_body_dist

one_body_dist = create_one_body_distribution(positions[:,:,:10], bins=40)



plt.plot(one_body_dist[:,-1])
plt.show()