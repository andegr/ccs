import numpy as np
from scipy.signal import correlate
from numba import njit, prange, int32, float64

from parameters import n_particles, friction_coef, kB, T, dimensions, L, xlo, xhi, r_cut, eps, sigma, dr

@njit
def sample_zeta(n_particles, dimensions=dimensions):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta

# Note: Parrallelization makes only sense for many particles, otherwise it is slower
@njit(parallel=True)
def Euler_Maruyama(positions, dt, hist, hist_distances=False):
    new_positions = np.empty_like(positions)

    # Sample independent zeta for each particle and dimension
    zeta = sample_zeta(n_particles, dimensions)
    prefactor_displ_vec = np.sqrt(2.0 * dt *kB*T / friction_coef)
    prefactor_ext_forces = dt/friction_coef
    F_ext, pbc_distances, hist = force_ext(positions, hist, hist_distances)

    for i in prange(n_particles):
        new_positions[i,:] = positions[i,:] + prefactor_ext_forces * F_ext[i,:] + prefactor_displ_vec * zeta[i,:]

        # Apply PBC to all dimensions 
        # NOTE: uncomment the following for MSD calculation
        for d in range(dimensions):
            # New: Correct Centered PBC Wrapping for [-L/2, L/2)
            shift =  np.rint(new_positions[i, d] / L)     # rounds to nearest integer, numba compatible
            new_positions[i, d] = new_positions[i, d] - L * shift
        
    return new_positions, pbc_distances, hist


@njit(parallel=True)
def force_ext(positions, hist, hist_distances):
    """Here positions = only the current positions, not the whole trajectory !
    , i.e shape(position) = (numb_part, dimensions)
    Retruns:
        forces:
        rij_distances
    """

    max_pairs = n_particles*(n_particles-1) // 2
    pbc_distances = np.zeros(max_pairs)     # number of all distances 
    distance_count = 0
    # forces of particles of each dimension, i.e. has shape numb_part, dimensions 
    forces = np.zeros(shape=positions.shape[0:2])     # numb_part, dimensions 
    
    
    for i in prange(positions.shape[0] - 1):    # run from 1st particle (i=0) to 2nd last particle (i=numb_part-1)
        for j in range(i+1, positions.shape[0]):# run from 2nd particle (j=1) to last particle (j=numb_part)
            
            rijx = pbc_distance(positions[i,0], positions[j,0], 0, L)    # x distance
            rijy = pbc_distance(positions[i,1], positions[j,1], 0, L)    # y
            rijz = pbc_distance(positions[i,2], positions[j,2], 0, L)    # z
            
            # r2 = rijx * rijx + rijy * rijy + rijz * rijz
            # r = np.sqrt(r2)

            r2 = rijx**2 + rijy**2 + rijz**2
            r = np.sqrt(r2) 

            if (r < L/2) and hist_distances:
                idx = int(r / dr)
                hist[idx] += 2

            if r < r_cut:
                '''Save absolute distance (for g(r) later)'''
                pbc_distances[distance_count] = r
                distance_count += 1
                '''calculate LJ-Interacion'''

                s = sigma / r               
                s2 = s*s                # why not only calculating s12 and s6 only and do it once outside the for loop ?
                s6 = s2 * s2 * s2
                s12 = s6 * s6


                LJ = 24 * eps * ( 2*(s12/r) - (s6/r))

                F_vec_over_r = LJ / r

                forces[i,0] -= F_vec_over_r * rijx
                forces[i,1] -= F_vec_over_r * rijy
                forces[i,2] -= F_vec_over_r * rijz

                forces[j,0] += F_vec_over_r * rijx  
                forces[j,1] += F_vec_over_r * rijy   
                forces[j,2] += F_vec_over_r * rijz    

    return forces, pbc_distances[:distance_count], hist   # return only relevant distances that are below cutoff



#---------------- Put it in the force loop, to avoid another loop over all pairs (very expensive)--------------


# @njit(parallel=False)
# def update_hist(hist, distances, dr):
#     """
#     Updates the raw histogram counts based on a set of pair distances.

#     Parameters:
#     hist (np.ndarray): The cumulative 1D histogram array (raw counts).
#     distances (np.ndarray): 1D array of pair distances collected in one timestep.
#     dr (float): The bin width (Delta r).
#     """
    
#     # Iterate over all distances collected in the current timestep
#     for i in range(distances.shape[0]):
#         r = distances[i]
        
#         k = int(r / dr)
        
#         # Check if the calculated index is within the bounds of the histogram array.
#         # This acts as a final safeguard, though r_cut/dr should define the hist size.
#         if k < hist.shape[0]:
#             # Increment the count for that bin
#             # We use += 1 as this is the standard way to count in a histogram
#             hist[k] += 1
#---------------------------------------------------------------------------------------------       

@njit  
def pbc_distance(xi, xj, xlo, xhi):
    """Calculation of shortest distance via Minimum Image Convention."""
    L = xhi - xlo  # L

    rij = xj - xi  # Calculate raw distance

    # Apply Minimum Image Convention
    if abs(rij) > 0.5 * L:
        # A Numba-friendly way to apply MIC is using the round function: - chatgbt says thats very slow
        rij = rij - L * np.round(rij / L)
        
    return rij



# def calculate_msd(trajectory_unwrapped):
#     """
#     Calculates the Mean Squared Displacement (MSD) for all particles over time.
    
#     Parameters:
#     trajectory_unwrapped (np.ndarray): Shape (N_particles, Dimensions, N_saved_steps)
    
#     Returns:
#     np.ndarray: MSD values for each saved timestep.
#     """
    
#     # Initial position is the first snapshot
#     initial_positions = trajectory_unwrapped[:, :, 0:1] # Shape (N, D, 1)

#     # Displacement vector at each step: R(t) - R(0)
#     # Broadcasting takes care of (N, D, T) - (N, D, 1)
#     displacement_vectors = trajectory_unwrapped - initial_positions 
    
#     # Squared displacement: (R(t) - R(0))^2
#     # Sum over dimensions (axis=1) -> Shape (N, T)
#     squared_displacement = np.sum(displacement_vectors**2, axis=1) 
    
#     # Mean Squared Displacement (MSD): Average over all particles (axis=0) -> Shape (T)
#     msd = np.mean(squared_displacement, axis=0) 
    
#     return msd

# def U_LJ(positions):
#     U = 0.0
#     n = positions.shape[0]
    
#     for i in range(n-1):
#         for j in range(i+1, n):
#             # minimum-image distance in 1D
#             rij = pbc_distance(positions[i,0], positions[j,0], 0, L)
            
#             r = abs(rij)
#             if r < r_cut:
#                 # U += 4*eps*( (sigma/r)**12 - (sigma/r)**6 )
#                 U += 4.0 * eps * ((sigma / r)**12 - (sigma / r)**6)
    
#     return U


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