import numpy as np
from scipy.signal import correlate
from numba import njit,prange

from parameters import n_particles, friction_coef, kB, T, dimensions, A,B,C, V_0

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
    prefactor_displ_vec = np.sqrt(2.0 * dt *kB*T / friction_coef)
    prefactor_ext_forces = -2*V_0/friction_coef

    for i in range(n_particles):

        # -dV/dx = -2*V_0 * (2*A*positions[i,:]**3 - B*positions[i,:])
        new_positions[i,:] = positions[i,:] + prefactor_ext_forces * (2*A*positions[i,:]**3 - B*positions[i,:]) * dt + prefactor_displ_vec * zeta[i,:]

    return new_positions
