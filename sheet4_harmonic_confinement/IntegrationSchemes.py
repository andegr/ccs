import numpy as np
from numba import njit,prange

from parameters import n_particles, friction_coef, K_H, dimensions

@njit
def sample_zeta(n_particles, dimensions=2):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta



@njit#(parallel=True)
def Euler_Maruyama(positions, dt, Analyze=False):
    new_positions = np.empty_like(positions)

    # Sample independent zeta for each particle and dimension
    zeta = sample_zeta(n_particles, dimensions)
    prefactor_displ_vec = np.sqrt(2.0 * dt / friction_coef) # assuming kB T = 1

    for i in range(n_particles):
        new_position = positions[i,:] - K_H * positions[i,:] + prefactor_displ_vec * zeta[i,:]
   
    displacement = prefactor_displ_vec * zeta
    return new_positions, displacement


