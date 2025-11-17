import numpy as np
from scipy.signal import correlate
from numba import njit,prange

from parameters import n_particles, friction_coef, K_H, dimensions

@njit
def sample_zeta(n_particles, dimensions=2):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta

# Note: Parrallelization makes only sense for many particles, otherwise it is slower

@njit#(parallel=True)
def Euler_Maruyama(positions, dt, Analyze=False):
    new_positions = np.empty_like(positions)

    # Sample independent zeta for each particle and dimension
    zeta = sample_zeta(n_particles, dimensions)
    prefactor_displ_vec = np.sqrt(2.0 * dt / friction_coef) # assuming kB T = 1
    prefactor_drift = -K_H/friction_coef

    for i in range(n_particles):
        new_positions[i,:] = positions[i,:] + prefactor_drift * positions[i,:] * dt + prefactor_displ_vec * zeta[i,:]

    displacement = prefactor_displ_vec * zeta
    return new_positions, displacement


@njit#(parallel=True)
def Ornstein_Uhlenbeck(positions, dt, Analyze=False):
    new_positions = np.empty_like(positions)

    # Sample independent zeta for each particle and dimension
    zeta = sample_zeta(n_particles, dimensions)
    lambdaa = K_H/friction_coef

    for i in range(n_particles):
        new_positions[i,:] = positions[i,:] * np.exp(-lambdaa * dt) + np.sqrt(1 / K_H*(1.0-np.exp(-2*lambdaa*dt))) * zeta[i,:]

    return new_positions