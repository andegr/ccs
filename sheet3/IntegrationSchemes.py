import numpy as np
from numba import njit,prange

from parameters import n_particles, friction_coef


@njit
def sample_eta(n_particles):
    eta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, 2))
    return eta


@njit(parallel = True)
def Euler_Maruyama(positions, dt, Analyze = False):
    new_positions = np.zeros_like(positions)

    eta = sample_eta(n_particles)

    for i in prange(n_particles):
        new_position = positions[i,:] + np.sqrt(dt/friction_coef) * eta[i,:]
        new_positions[i,:] = new_position   

    return new_positions


    



