import numpy as np
from scipy.signal import correlate
from numba import njit,prange

from parameters import n_particles, friction_coef, dimensions, x_R

@njit
def sample_zeta(n_particles, dimensions=2):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta



@njit#(parallel=True)
def Euler_Maruyama(positions, absorption_counter, dt, Analyze=False):
    new_positions = np.empty_like(positions)

    zeta = sample_zeta(n_particles, dimensions)
    prefactor_displ_vec = np.sqrt(2.0 * dt / friction_coef) # assuming kB T = 1

    for i in range(n_particles):
        new_positions[i,:] = positions[i,:] + prefactor_displ_vec * zeta[i,:]

        if new_positions[i,:] > x_R:
            new_positions[i,:] = 0
            absorption_counter += 1

        elif new_positions[i,:] < 0:
            new_positions[i,:] = -new_positions[i,:]

    displacement = prefactor_displ_vec * zeta
    return new_positions, displacement, absorption_counter
