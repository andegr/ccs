import numpy as np
import numba
from numba import njit, prange, int32, float64
from parameters import MDSimulationParameters

@njit
def sample_zeta(n_particles, dimensions=2):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta

# Note: Parrallelization makes only sense for many particles, otherwise it is slower


@njit(parallel=True)
def Euler_Maruyama(positions, orientations,
                   dimensions, L, r_cut, eps, sigma,
                   dt, kB, T, Dt, Dr, v0, walls):
    """
    One Euler–Maruyama step for free Active Brownian Particles (2D).
    """

    n_particles = positions.shape[0]

    new_positions = np.empty_like(positions)
    new_orientations = np.empty_like(orientations)

    # Remember: Effective propulsion speed v0 = beta * Dt * F

    zeta_r = sample_zeta(n_particles, dimensions)
    prefactor_r = np.sqrt(2.0 * Dt * dt)

    zeta_theta = np.random.randn(n_particles)
    prefactor_theta = np.sqrt(2.0 * Dr * dt)

    for i in prange(n_particles):

        # orientation update
        theta_new = orientations[i] + prefactor_theta * zeta_theta[i]
        new_orientations[i] = theta_new % (2*np.pi)

        # propulsion direction
        nx = np.cos(theta_new)
        ny = np.sin(theta_new)

        # position update
        new_positions[i, 0] = positions[i, 0] + v0 * nx * dt + prefactor_r * zeta_r[i, 0]
        new_positions[i, 1] = positions[i, 1] + v0 * ny * dt + prefactor_r * zeta_r[i, 1]

        if walls:
            if (new_positions[i, 0] < 0) or (new_positions[i, 0] > L):
                new_positions[i, 0] = positions[i, 0]
                # not resetting the y positions as well is like realizing half a step
                # and I think it could cause unexpected behaviour
                new_positions[i, 1] = positions[i, 1]


        # --- periodic boundary conditions ---
        # commented out for MSD calculation
        # new_positions[i, 0] %= L
        # new_positions[i, 1] %= L

    return new_positions, new_orientations





@njit(parallel=True)
def force_ext(positions, orientations, 
              dr, dimensions, L, r_cut, eps, sigma, dt, kB, T):
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
def update_histogram_all_pairs(positions, hist, dr, L):
    n_particles = positions.shape[0]
    
    # Loop over all unique pairs (i < j)
    for i in range(n_particles):
        for j in range(i + 1, n_particles): 
            
            rijx = pbc_distance(positions[i, 0], positions[j, 0], 0, L)
            rijy = pbc_distance(positions[i, 1], positions[j, 1], 0, L)
            r2 = rijx**2 + rijy**2
            
            if r2 < (L/2)**2: 
                r = np.sqrt(r2)
                hist_idx = int(r / dr)
                hist[hist_idx] += 2 
                    
    return hist


@njit  
def pbc_distance(xi, xj, xlo, xhi):
    """Calculation of shortest distance via Minimum Image Convention."""
    L = xhi - xlo  # L

    rij = xj - xi  # Calculate raw distance

    # Apply Minimum Image Convention
    if abs(rij) > 0.5 * L:
        # A Numba-friendly way to apply MIC is using the round function: - chatgbt says thats very slow -->jonas: changed it to floor
        rij = rij - L * np.round(rij / L)
        
    return rij


