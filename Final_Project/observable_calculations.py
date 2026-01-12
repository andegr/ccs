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


def one_particle_density(positions_trajectory, numb_bins):
    x_positions_trajectory = positions_trajectory[:,0,:]
    x_all = x_positions_trajectory.ravel()
    density, bins = np.histogram(x_all, bins=numb_bins, density=True)
    bin_centers = 0.5*(bins[:-1] + bins[1:])

    return density, bin_centers


def average_one_particle_density(traj_list, n_bins=200, x_range=None, direction="x"):
    """
    Estimate one-particle density rho(x) from multiple runs.
    """
    rho_runs = []

    for traj in traj_list:
        # take x component, flatten over particles and time 
        # since every particle and timestep contributes equally
        if direction == "x":
            x_all = traj[:, 0, :].ravel()

        elif direction == "y":
            x_all = traj[:, 1, :].ravel()

        else:
            raise "wrong direction chosen"
        # If range is not given in np.histogram, it uses [min(xdata),max(xdata)]
        # --> important for multiple runs such that always the same bin edges are chosen 
        rho, bins = np.histogram(x_all, bins=n_bins, range=x_range, density=True)
        rho_runs.append(rho)

    rho_mean = np.mean(np.stack(rho_runs, axis=0), axis=0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, rho_mean