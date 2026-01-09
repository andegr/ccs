import numpy as np
from numba import njit, prange


@njit(parallel=True)
def orientation_autocorrelation(orientations): 
    """
    Fast orientation autocorrelation using single time origin. ( but therefore little statistics, always just one sample)

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


@njit
def orientation_autocorrelation_averaged(orientations, blocks=200): 
    """
    

    orientations: (N, 2, T)
    """
    N, _, T = orientations.shape

    t_block =  T // blocks

    
    C = np.zeros(t_block)

    for b in range(blocks):
        for t in range(t_block):
            s = 0.0
            for i in range(N):
                s += (
                    orientations[i, 0, t + b*t_block] * orientations[i, 0, 0+ b*t_block]
                    + orientations[i, 1, t+ b*t_block] * orientations[i, 1, 0+ b*t_block]
                )
            C[t] += s

    return C / (N*blocks)


    
def msd_numerical(trajectory_unwrapped):
    # trajectory_unwrapped: (N, D, T)
    initial_positions = trajectory_unwrapped[:, :, 0:1]  # (N, D, 1)  <-- key change

    displacement_vectors = trajectory_unwrapped - initial_positions  # (N, D, T)
    squared_displacement = np.sum(displacement_vectors**2, axis=1)   # (N, T)
    msd = np.mean(squared_displacement, axis=0)                      # (T,)
    return msd


def calculate_average_msd(traj_list):
    msds = []
    for traj in traj_list:
        run_msd = msd_numerical(traj)
        msds.append(run_msd)

    msds = np.array(msds)
    mean_msd = np.mean(msds, axis=0)

    return mean_msd



def msd_theory(t, v0, Dt, Dr):
    a = 4*Dt*t
    b = 2*v0**2/Dr
    c = t - (1- np.exp(-Dr*t) / Dr)

    return a + b * c 

