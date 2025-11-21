import numpy as np
from numba import njit, prange

@njit(parallel=True)
def autocorrelation(x, min_sample_size=10):
    n_x = len(x)

    E_x2 = np.mean(x * x)

    autocorr = np.zeros(int(n_x / min_sample_size))

    for i in prange(len(autocorr)):
        # lag in samples
        lag = i
        
        # number of valid pairs (x[j], x[j+lag])
        n_pairs = n_x - lag
        
        # compute sum of products
        s = np.sum(x[:n_pairs] * x[lag:n_x])
        
        autocorr[i] = s / (n_pairs * E_x2)

    return autocorr
