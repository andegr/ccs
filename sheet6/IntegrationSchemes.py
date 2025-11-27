import numpy as np
from scipy.signal import correlate
from numba import njit,prange

from parameters import n_particles, friction_coef, kB, T, dimensions, dimensions_task1, L, xlo, xhi, r_cut, eps, sigma

@njit
def sample_zeta(n_particles, dimensions=2):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta

# Note: Parrallelization makes only sense for many particles, otherwise it is slower

@njit(parallel=True)
def Euler_Maruyama(positions, dt, Analyze=False):
    new_positions = np.empty_like(positions)

    # Sample independent zeta for each particle and dimension
    zeta = sample_zeta(n_particles, dimensions)
    prefactor_displ_vec = np.sqrt(2.0 * dt *kB*T / friction_coef)
    prefactor_ext_forces = dt/friction_coef
    F_ext = force_ext(positions)

    for i in range(n_particles):
        new_positions[i,:] = positions[i,:] + prefactor_ext_forces * F_ext[i,:] + prefactor_displ_vec * zeta[i,:]

        # apply periodic boundary conditions such that particles don't leave box
        new_positions[i,0] = new_positions[i,0] % L     
        
    return new_positions


@njit(parallel=True)
def force_ext(positions):
    """Here positions = only the current positions, not the whole trajectory !
    , i.e shape(position) = (numb_part, dimensions)
    """
    # forces of particles of each dimension, i.e. has shape numb_part, dimensions 
    forces = np.zeros(shape=positions.shape[0:2])     # numb_part, dimensions 
    
    for i in prange(positions.shape[0] - 1):    # run from 1st particle (i=0) to 2nd last particle (i=numb_part-1)
        for j in range(i+1, positions.shape[0]):# run from 2nd particle (j=1) to last particle (j=numb_part)
            
            rijx = pbc_distance(positions[i,0], positions[j,0], 0, L)    # x distance
            # rijy = pbc_distance(positions[i,1], positions[j,1], 0, L)    # y
            # rijz = pbc_distance(positions[i,2], positions[j,2], 0, L)    # z
            
            # r2 = rijx * rijx + rijy * rijy + rijz * rijz
            # r = np.sqrt(r2)

            r = rijx

            if r < r_cut:
                '''calculate LJ-Interacion'''
                LJ = 24 * eps * ( 2*(sigma**12)/r**14 - (sigma**6)/r**8 )

                forces[i,0] -= LJ * rijx        # add x comp. force of j-th particle acting on i-th particle
                forces[j,0] += LJ * rijx        # add x comp. force of i-th particle acting on j-th particle

    return forces



@njit  
def pbc_distance(xi, xj, xlo, xhi):
    """Calculation of distance of particles i and j with taking into account the shortest
    distance via Periodic Boundary Conditions (pbc) for one dimension
    
    Returns:
    ---------
    rij: minimum distance"""
    
    l = xhi-xlo
    
    xi = xi % l     # let i-th particle reenter from other side x got bigger than L
    xj = xj % l     # let j-th particle reenter from other side x got bigger than L

    rij = xj - xi  
    if abs(rij) > 0.5*l:
        rij = rij - np.sign(rij) * l 
        
    return rij


@njit
def pbc_distance_array(xi_arr, xj_arr, xlo, xhi):
    n = xi_arr.shape[0]
    rij_arr = np.empty(n)

    for k in range(n):
        rij_arr[k] = pbc_distance(xi_arr[k], xj_arr[k], xlo, xhi)

    return rij_arr


def U_LJ(positions):
    U = 0.0
    n = positions.shape[0]
    
    for i in range(n-1):
        for j in range(i+1, n):
            # minimum-image distance in 1D
            rij = pbc_distance(positions[i,0], positions[j,0], 0, L)
            
            r = abs(rij)
            if r < r_cut:
                # U += 4*eps*( (sigma/r)**12 - (sigma/r)**6 )
                U += 4.0 * eps * ((sigma / r)**12 - (sigma / r)**6)
    
    return U


# def g_of_r_hist_2D(positions):
#     """Calculates histogramm of current timestep of """
#     n = positions.shape[0]      # number of particles

#     num_of_bins = 250
#     dr = (L/2) // 250
#     dr_arr = np.linspace(dr/2, L/2, num=num_of_bins)  # ???

#     rij_arr = np.zeros(n*(n-1))       # ???

#     for i in range(n-1):
#         for j in range(i, n):
#             rij_x = pbc_distance(positions[i,0], positions[j,0], 0, L)
#             rij_y = pbc_distance(positions[i,1], positions[j,1], 0, L)

#             rij = np.sqrt(rij_x**2 + rij_y**2)

#             rij_arr[i+j] = rij  # ???