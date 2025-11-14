import numpy as np
from scipy.signal import correlate
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
    # prefactor_rdm_noise = 1 / K_H               # actually kBT/K_H

    for i in range(n_particles):
        new_positions[i,:] = positions[i,:] * np.exp(-lambdaa * dt) + np.sqrt(1 / K_H*(1.0-np.exp(-2*lambdaa*dt))) * zeta[i,:]

    return new_positions


# def AutoCorrelation(data):
#     """calculates the ACF of every particle and every dimension 
#     and averages over them to properly normalize it
    
#     Parameters:
    
#     data: array of particles, dimensions and timesteps
#     return:
#     acorr: normalized array of the autocorrelation"""

#     n_particles, n_dim, n_time = data.shape
#     acorr_all = np.zeros((n_particles, n_dim, n_time))

#     for p in range(n_particles):
#         for d in range(n_dim):
#             ndata = data[p, d, :] - np.mean(data[p, d, :])
#             var = np.var(ndata)
#             # using correlate from scipy instead of np.correlate to use faster fft algorithm
#             acorr = correlate(ndata, ndata, mode='full')[n_time-1:]         
#             # acorr = np.correlate(ndata, ndata, mode='full')[n_time-1:]    # testing it showed no diff tho     
#             # From numpy documentation:
#             # mode is ‘full’. This returns the convolution at each point of overlap, with an output shape of (N+M-1,) 
#             # where N is the shape of the 1st and M of the 2nd array, so here: acorr of shape (2N-1,)  
#             # At the end-points of the convolution, the signals do not overlap completely, and boundary effects may be seen.
#             normalization = np.arange(n_time, 0, -1)        # creates arr = [N, N-1,..., 1] to properly normalize each step
#             acorr = acorr / normalization / var
#             acorr_all[p, d, :] = acorr

#     # summing over axes of particles and dimensions and deviding by their lengths
#     acorr_mean = np.mean(acorr_all, axis=(0,1))     
#     return acorr_mean