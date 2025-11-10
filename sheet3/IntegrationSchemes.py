import numpy as np
from numba import njit,prange

from parameters import n_particles, friction_coef


# @njit
# def sample_zeta(n_particles):
#     zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, 2))
#     return zeta


# @njit(parallel = True)
# def Euler_Maruyama(positions, dt, Analyze = False):
#     new_positions = np.zeros_like(positions)

#     zeta = sample_zeta(n_particles)

#     for i in prange(n_particles):
# !!!!!!!!! Achtung !!!!!!!!!!!!
# --> hier werden glaube ich die Zetas alle mit 
#         new_position = positions[i,:] + np.sqrt(dt/friction_coef) * zeta[i,:]       # fehlt hier nicht eine sqrt(2) ?
#                                                                                     # siehe eq. (1) auf Blatt
                                                                                      
#         new_positions[i,:] = new_position   

#     return new_positions

@njit
def sample_zeta(n_particles, dimensions=2):
    # Numba-compatible random number generation without prange
    # since in Numba docs it says: "Parallel random number generation (np.random.*) inside prange loops is not supported."
    # for i in range(n_particles):
    #     for d in range(dimensions):
    #         zeta[i, d] = np.random.normal(0.0, 1.0)
    # return zeta
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta



@njit(parallel=True)
def Euler_Maruyama(positions, dt, Analyze=False):
    new_positions = np.empty_like(positions)

    # Sample independent zeta for each particle and dimension
    zeta = sample_zeta(n_particles, 2)
    prefactor_displ_vec = np.sqrt(2.0 * dt / friction_coef) # assuming kB T = 1

    for i in prange(n_particles):
        new_position = positions[i,:] + prefactor_displ_vec * zeta[i,:]
    # for i in prange(n_particles):
    #     for d in range(dimensions):
    #         new_positions[i, d] = current_positions[i, d] + prefactor_displ_vec * zeta[i, d]

    displacement = prefactor_displ_vec * zeta
    return new_positions, displacement


