import numpy as np
from numba import njit, prange
from IntegrationSchemes import pbc_distance


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



# -------- Find Cluster functions --------- # 
import numpy as np
from numba import njit
import numpy as np
from numba import njit

@njit
def find_root(parent, i):
    """
    Return the representative ("root") of the set that particle i belongs to.

    parent[] stores a forest of trees:
      - If parent[x] == x, then x is a root (representative).
      - Otherwise parent[x] points "up" towards the root.

    This uses *path compression*:
      while walking up to the root, we make nodes point closer to the root.
      That makes future calls much faster.
    """
    while parent[i] != i:
        # Make i jump one level up (compress path)
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit
def merge_sets(parent, set_size, a, b):
    """
    Merge the sets (clusters) that contain particle a and particle b.

    Steps:
      1) Find roots ra and rb (representatives of the clusters).
      2) If they are already same root -> already in same cluster -> nothing to do.
      3) Otherwise, attach the smaller cluster under the bigger cluster
         (this is called union-by-size), and update the size of the new root.
    """
    ra = find_root(parent, a)
    rb = find_root(parent, b)

    if ra == rb:
        return  # already same cluster

    # Union-by-size: attach smaller tree under larger tree's root
    if set_size[ra] < set_size[rb]:
        ra, rb = rb, ra  # swap so ra is the bigger one
    
    parent[rb] = ra              # rb now points to ra => clusters merged
    set_size[ra] += set_size[rb] # update size of merged cluster (root ra)

@njit
def find_cluster_stats(positions, L, r_cut_clus):
    """
    Compute simple cluster statistics from one snapshot.

    Cluster definition:
      - particles i and j are "connected" if their (PBC) distance rij < r_cut_clus
      - clusters = connected components of this connectivity graph

    Returns
    -------
    max_cluster_size : int
        Size of the largest cluster (includes monomers -> minimum is 1).
    n_clusters : int
        Number of clusters with size >= 2 (i.e., excluding monomers).
    n_monomers : int
        Number of monomers.
    mean_mass_weighted : float
        Mass-weighted mean cluster size: sum_c s_c^2 / N
        (interpretation: expected cluster size of a randomly chosen particle)
    """
    N = positions.shape[0]
    r2_cut = r_cut_clus * r_cut_clus

    # Disjoint-set (union–find) structure:
    # parent[i] points to parent of i in a tree; roots have parent[root] == root.
    parent = np.empty(N, dtype=np.int64)
    # set_size[root] stores cluster size for roots (only reliable for roots during merging)
    set_size = np.ones(N, dtype=np.int64)

    for i in range(N):
        parent[i] = i

    # --- Build connectivity by merging sets when i and j are neighbors ---
    for i in range(N - 1):
        xi = positions[i, 0]
        yi = positions[i, 1]
        for j in range(i + 1, N):
            rijx = pbc_distance(xi, positions[j, 0], 0.0, L)
            rijy = pbc_distance(yi, positions[j, 1], 0.0, L)
            r2 = rijx*rijx + rijy*rijy
            if r2 < r2_cut:
                merge_sets(parent, set_size, i, j)

    # --- Compress parents so every node points (close) to its root ---
    for i in range(N):
        parent[i] = find_root(parent, i)

    # --- Count final cluster sizes by root (robust) ---
    sizes_by_root = np.zeros(N, dtype=np.int64)
    for i in range(N):
        sizes_by_root[parent[i]] += 1

    # --- Compute stats ---
    max_cluster_size = 1
    n_clusters = 0      # exclude monomers (size 1)
    n_monomers = 0
    sum_s2 = 0.0

    for r in range(N):
        s = sizes_by_root[r]
        if s > 0:  # r is a root that actually has members
            if s > max_cluster_size:
                max_cluster_size = s
            if s >= 2:
                n_clusters += 1
            elif s == 1:
                n_monomers += 1
            sum_s2 += s * s

    mean_mass_weighted = sum_s2 / N

    return max_cluster_size, n_clusters, n_monomers, mean_mass_weighted


def cluster_stats_traj(positions_trajectory, L, r_cut_clus):
    # positions_trajectory: (N, D, T)
    T = positions_trajectory.shape[2]

    # one value per time step
    smax_arr = np.zeros(T, dtype=np.int64)
    nclus_arr = np.zeros(T, dtype=np.int64)
    n_monomers_arr = np.zeros(T, dtype=np.int64)
    smeanw_arr = np.zeros(T, dtype=np.float64)

    for i in range(T):
        smax, nclus, n_monomers, smeanw = find_cluster_stats(positions_trajectory[:, :, i], L, r_cut_clus)
        smax_arr[i] = smax
        nclus_arr[i] = nclus
        n_monomers_arr[i] = n_monomers
        smeanw_arr[i] = smeanw

    return smax_arr, nclus_arr, n_monomers_arr, smeanw_arr