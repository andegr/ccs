import numpy as np
from scipy.signal import correlate
from numba import njit,prange

from parameters import n_particles, friction_coef, kB, T, dimensions, dimensions_task1, L, xlo, xhi, r_cut, eps, sigma

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
    prefactor_ext_forces = -dt/friction_coef

    for i in range(n_particles):

        new_positions[i,:] = positions[i,:] + prefactor_ext_forces * (2*A*positions[i,:]**3 - B*positions[i,:]) *  + prefactor_displ_vec * zeta[i,:]

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
                LJ = 24 * eps * ( sigma**6 / r**8 - 2*sigma**12/ r**14)    # LJ "prefactor"

                forces[i,0] -= LJ * rijx        # add x comp. force of j-th particle acting on i-th particle
                forces[j,0] += LJ * rijx        # add x comp. force of i-th particle acting on j-th particle

    return positions, forces



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

def U_LJ(positions):
    U = 0.0
    n = positions.shape[0]
    
    for i in range(n-1):
        for j in range(i+1, n):
            # minimum-image distance in 1D
            rij = pbc_distance(positions[i,0], positions[j,0], 0, L)
            
            r = abs(rij)
            if r < r_cut:
                U += 4*eps*( (sigma/r)**12 - (sigma/r)**6 )
    
    return U


# def pbc_distance_np(xi, xj, xlo, xhi):
    
#     l = xhi-xlo
    
#     xi = np.mod(xi, l)
#     xj = np.mod(xj, l)

#     np.where(rij = xj - xi)
#     rij = rij - np.sign(rij) * l 

#     return rij