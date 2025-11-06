import numpy as np
from numba import njit,prange

from parameters import n_particles, friction_coef


@njit
def sample_zeta(n_particles):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, 2))
    return zeta


@njit(parallel = True)
def Euler_Maruyama(positions, dt, Analyze = False):
    new_positions = np.zeros_like(positions)

    zeta = sample_zeta(n_particles)

    for i in prange(n_particles):
        new_position = positions[i,:] + np.sqrt(dt/friction_coef) * zeta[i,:]
        new_positions[i,:] = new_position   

    return new_positions


    



