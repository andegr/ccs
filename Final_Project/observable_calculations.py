import numpy as np
from numba import njit, prange


@njit(parallel=True)
def orientation_autocorrelation(orientations):      # faster way to calculate it
    """
    Fast orientation autocorrelation using single time origin.

    orientations: (N, 2, T)
    """
    N, _, T = orientations.shape
    C = np.zeros(T)

    for t in prange(T):
        s = 0.0
        for i in range(N):
            s += (
                orientations[i, 0, t] * orientations[i, 0, 0]
                + orientations[i, 1, t] * orientations[i, 1, 0]
            )
        C[t] = s / N

    return C


def calculate_msd(trajectory_unwrapped):
    """
    Calculates the Mean Squared Displacement (MSD) for all particles over time.
    
    Parameters:
    trajectory_unwrapped (np.ndarray): Shape (N_particles, Dimensions, N_saved_steps)
    
    Returns:
    np.ndarray: MSD values for each saved timestep.
    """
    
def calculate_msd(trajectory_unwrapped):
    # trajectory_unwrapped: (N, D, T)
    initial_positions = trajectory_unwrapped[:, :, 0:1]  # (N, D, 1)  <-- key change

    displacement_vectors = trajectory_unwrapped - initial_positions  # (N, D, T)
    squared_displacement = np.sum(displacement_vectors**2, axis=1)   # (N, T)
    msd = np.mean(squared_displacement, axis=0)                      # (T,)
    return msd