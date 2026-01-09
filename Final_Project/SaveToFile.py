import numpy as np
# from parameters import MDSimulationParameters
from time import time
from helpers import build_traj_fname
# Assuming MDSimulationParameters is imported from parameters

# ----------------------------Ovito Trajectory:----------------------------#

def save_OVITO(
    positions: np.ndarray,
    thetas: np.ndarray,
    parameters,
    file_name: str = "trajectory.txt",
    save_interval: int = 1,
) -> None:
    """
    Saves a LAMMPS-dump style trajectory readable by OVITO, including orientations.

    positions: (N, D, T)
    thetas:    (N, T)  (orientation angle per particle per saved frame)

    The orientation is stored as a vector (cosθ, sinθ, 0) in columns vx vy vz.
    OVITO can display this via the Vector Display modifier using the Velocity property.
    """
    dimensions = parameters.dimensions
    N, D_check, T = positions.shape
    if D_check != dimensions:
        raise ValueError(f"Positions have D={D_check} but parameters.dimensions={dimensions}")

    if thetas.shape != (N, T):
        raise ValueError(f"thetas must have shape (N,T)=({N},{T}), got {thetas.shape}")

    # Build unit orientation vectors from theta: v = (cosθ, sinθ, 0)
    vx_all = np.cos(thetas)
    vy_all = np.sin(thetas)
    vz_all = np.zeros_like(thetas)

    # Metadata
    xlo, xhi = parameters.xlo, parameters.xhi
    ylo, yhi = parameters.ylo, parameters.yhi
    zlo, zhi = parameters.zlo, parameters.zhi

    # Keep your original convention:
    dt_save = parameters.n_save

    with open(file_name, "w") as f:
        for t in range(0, T, save_interval):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{t * dt_save}\n")

            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{N}\n")

            f.write("ITEM: BOX BOUNDS\n")
            f.write(f"{xlo} {xhi} xlo xhi\n")
            f.write(f"{ylo} {yhi} ylo yhi\n")
            f.write(f"{zlo} {zhi} zlo zhi\n")

            # Store orientation vector as vx vy vz (recognized by OVITO as Velocity)
            f.write("ITEM: ATOMS id type x y z vx vy vz\n")

            for i in range(N):
                x = positions[i, 0, t]
                y = positions[i, 1, t] if dimensions > 1 else 0.0
                z = positions[i, 2, t] if dimensions > 2 else 0.0

                vx = vx_all[i, t]
                vy = vy_all[i, t]
                vz = vz_all[i, t]

                f.write(f"{i} {i} {x} {y} {z} {vx} {vy} {vz}\n")

    print(f"Trajectory data saved in file: {file_name}")


def load_OVITO(
    file_name: str,
    parameters,
    return_thetas: bool = False,
):
    """
    Loads a trajectory written by save_OVITO().

    Returns:
      positions: (N, D, T)
      optionally thetas: (N, T) reconstructed from vx,vy via atan2(vy, vx)
    """
    dimensions = parameters.dimensions

    with open(file_name, "r") as f:
        lines = f.readlines()

    positions_list = []
    thetas_list = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            i += 2  # skip timestep value line

            if lines[i].strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Unexpected dump format: missing 'ITEM: NUMBER OF ATOMS'")
            num_atoms = int(lines[i + 1].strip())
            i += 2

            if not lines[i].startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: missing 'ITEM: BOX BOUNDS'")
            i += 4  # header + 3 bounds lines

            atoms_header = lines[i].strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: missing 'ITEM: ATOMS'")

            cols = atoms_header.split()[2:]  # after "ITEM: ATOMS"

            # required position cols
            try:
                ix = cols.index("x")
                iy = cols.index("y") if "y" in cols else None
                iz = cols.index("z") if "z" in cols else None
            except ValueError as e:
                raise ValueError(f"Dump file missing x/y/z columns: {atoms_header}") from e

            # optional orientation-vector cols
            has_v = all(c in cols for c in ("vx", "vy", "vz"))
            if has_v:
                ivx, ivy = cols.index("vx"), cols.index("vy")

            i += 1  # first atom line

            pos_t = np.zeros((num_atoms, dimensions), dtype=float)
            theta_t = np.zeros((num_atoms,), dtype=float) if has_v else None

            for j in range(num_atoms):
                parts = lines[i + j].strip().split()

                pos_t[j, 0] = float(parts[ix])
                if dimensions > 1 and iy is not None:
                    pos_t[j, 1] = float(parts[iy])
                if dimensions > 2 and iz is not None:
                    pos_t[j, 2] = float(parts[iz])

                if has_v:
                    vx = float(parts[ivx])
                    vy = float(parts[ivy])
                    theta_t[j] = np.arctan2(vy, vx)

            positions_list.append(pos_t)
            if has_v:
                thetas_list.append(theta_t)

            i += num_atoms
        else:
            i += 1

    positions = np.stack(positions_list, axis=2)  # (N, D, T)

    if return_thetas:
        if len(thetas_list) == 0:
            raise ValueError("No vx/vy columns found in file, cannot reconstruct thetas.")
        thetas = np.stack(thetas_list, axis=1)  # (N, T)
        return positions, thetas

    return positions


#----------------------------Less Overhead Trajectory:----------------------------#
def save_positions_txt(positions: np.ndarray, parameters, filename: str) -> None:
    """
    Saves positions of shape (N_particles, dimensions, N_timesteps)
    to a txt file with minimal overhead.

    Parameters:
    - positions (np.ndarray): Array of particle positions.
    - parameters: An instance of MDSimulationParameters to get the dimensions.
    - filename (str): Output file name.
    """
    N, D, T = positions.shape

    if D != parameters.dimensions:
        raise ValueError(f"Dimensions in positions array ({D}) do not match parameters ({parameters.dimensions}).")

    # Flatten from (N, D, T) -> (T, N*D)
    flat = positions.transpose(2, 0, 1).reshape(T, N * D)

    # Write header
    with open(filename, "w") as f:
        # Save N, D, T from the array shape, not parameters, as they are inherent to the data
        f.write(f"{N} {D} {T}\n")

    # Append numeric data manually
    with open(filename, "a") as f:
        np.savetxt(f, flat, fmt="%.12g", delimiter=" ")

    print(f"Saved positions to {filename}")
    
def load_positions_txt(filename: str) -> np.ndarray:
    """
    Load positions saved with save_positions_txt() back into
    an array of shape (N_particles, dimensions, N_timesteps).
    
    Note: This function does not require MDSimulationParameters as 
    N, D, and T are read from the file's header.

    Parameters:
    - filename (str): Input file name.

    Returns:
    - np.ndarray: Array of particle positions (N_particles, dimensions, N_timesteps).
    """
    with open(filename, "r") as f:
        header = f.readline().strip().split()
        N, D, T = map(int, header)

    # Load the rest: shape (T, N*D)
    flat = np.loadtxt(filename, skiprows=1)

    # reshape back
    positions = flat.reshape(T, N, D).transpose(1, 2, 0)

    print(f"Loaded positions from {filename}")
    return positions


def load_runs(n_particles, t_sim, dt, v0, Dt, Dr, n_runs):
    '''
    Loads multiple runs for given parameters and returns
    the according time array and the runs in a list.
    
    :param n_particles: Description
    :param t_sim: Description
    :param dt: Description
    :param v0: Description
    :param Dt: Description
    :param Dr: Description
    :param n_runs: number of runs to load, loads all run_ids from 0 to n_runs -1.
    '''
    traj_list = []

    start_time = time()
    print("Started loading trajectories...")

    for run_id in range(n_runs):

        fname = "outputs/" + build_traj_fname(
            "traj_positions",
            n_particles,
            t_sim,
            dt,
            v0,
            Dt,
            Dr,
            run_id,
        ) 

        traj = load_positions_txt(filename=fname)

        traj_list.append(traj)

    print("All runs loaded. Time elapsed:", time()-start_time, "s")


    n_steps = traj.shape[-1]  
    dt_saved = dt * np.rint(t_sim / dt/ n_steps)
    time_arr = np.arange(n_steps) * dt_saved 

    return time_arr, traj_list




def save_orientations_txt(thetas: np.ndarray, filename: str) -> None:
    """
    Saves orientations as (cos(theta), sin(theta)) with minimal overhead.

    Parameters:
    - thetas (np.ndarray): Array of shape (N_particles, N_timesteps)
    - filename (str): Output file name
    """
    if thetas.ndim != 2:
        raise ValueError("thetas must have shape (N_particles, N_timesteps)")

    N, T = thetas.shape
    D = 2  # cos(theta), sin(theta)

    # Compute orientation vectors
    # Shape: (N, 2, T)
    orientations = np.empty((N, D, T), dtype=thetas.dtype)
    orientations[:, 0, :] = np.cos(thetas)
    orientations[:, 1, :] = np.sin(thetas)

    # Flatten from (N, 2, T) -> (T, N*2)
    flat = orientations.transpose(2, 0, 1).reshape(T, N * D)

    # Write header
    with open(filename, "w") as f:
        f.write(f"{N} {D} {T}\n")

    # Append numeric data
    with open(filename, "a") as f:
        np.savetxt(f, flat, fmt="%.8g", delimiter=" ")

    print(f"Saved orientations (cosθ, sinθ) to {filename}")


def load_orientations_txt(filename: str) -> np.ndarray:
    """
    Load orientations saved with save_orientations_txt() back into
    an array of shape (N_particles, 2, N_timesteps),
    where axis 1 is (cos(theta), sin(theta)).

    Parameters:
    - filename (str): Input file name.

    Returns:
    - np.ndarray: Orientation vectors (N, 2, T)
    """
    with open(filename, "r") as f:
        header = f.readline().strip().split()
        N, D, T = map(int, header)

    if D != 2:
        raise ValueError(f"Expected D=2 (cosθ, sinθ), got D={D}")

    # Load numeric data: shape (T, N*2)
    flat = np.loadtxt(filename, skiprows=1)

    # Reshape back: (T, N, 2) -> (N, 2, T)
    orientations = flat.reshape(T, N, 2).transpose(1, 2, 0)

    print(f"Loaded orientations (cosθ, sinθ) from {filename}")
    return orientations

#----------------------------Save Observables----------------------------#

def save_timesteps_and_observable(
    timesteps: np.ndarray, 
    observable: np.ndarray, 
    filename: str,
    # Parameters object is not needed here
) -> None:
    """
    Writes timesteps and corresponding energies/observable to a text file with two columns.

    Parameters:
    - timesteps (np.ndarray): Array of time steps (integers).
    - observable (np.ndarray): Array of observable values.
    - filename (str): Output text file name.
    """

    if len(timesteps) != len(observable):
        raise ValueError("Timesteps and observable arrays must have the same length.")

    data = np.column_stack((timesteps, observable))
    header = "TimeStep\tObservable"
    np.savetxt(filename, data, header=header, fmt='%d\t%.6f', delimiter='\t')
    
    
def load_timesteps_and_observable(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads timesteps (as integers) and observable values (as floats) from a text file with two columns.
    
    Note: This function does not require MDSimulationParameters.

    Parameters:
    - filename (str): Input text file name.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Two arrays, (timesteps: int array, observable: float array).
    """
    data = np.loadtxt(filename, skiprows=1)  
    timesteps = data[:, 0].astype(int)      
    observable = data[:, 1]                
    return timesteps, observable  


#--------------------save g(r)---------------------

def save_hist(
    hist_normalized: np.ndarray, 
    dr: float, 
    filename: str,
    # Parameters object is not needed here
) -> None:
    """
    Save a normalized histogram to a text file, including the bin width in the header.
    """
    header = f"bin_width = {dr}"
    np.savetxt(filename, hist_normalized, header=header)


def load_hist(filename: str) -> tuple[np.ndarray, float]:
    """
    Load a histogram saved with `save_hist`, returning both the array and the bin width.
    
    Note: This function does not require MDSimulationParameters.

    Parameters:
    - filename (str): Input file name.

    Returns
    -------
    tuple[np.ndarray, float]:
        hist : np.ndarray
            The histogram values.
        dr : float
            The bin width extracted from the header.
    """
    dr = None # Initialize dr to handle case where header is missing

    # Read header line
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                # Expected format: "# bin_width = <value>"
                if "bin_width" in line:
                    # Safely extract dr value after '=' and strip whitespace
                    try:
                        dr = float(line.split("=")[1].strip())
                    except:
                        # Handle potential parsing error if format is unexpected
                        print(f"Warning: Could not parse 'dr' from header line: {line.strip()}")
                continue
            else:
                break # Stop after header
        
    # Load data normally, skipping the header line(s)
    # np.loadtxt is robust and handles commented lines itself, but using the file object
    # after reading the header is less straightforward. Sticking to the original
    # logic which assumes the file pointer is past the header (or reloading it).
    
    # Reload file for np.loadtxt to read from the beginning, skipping the correct number of lines
    # The number of header lines (starting with #) can vary.
    
    # Count header lines
    skiprows = 0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                skiprows += 1
            else:
                break
                
    hist = np.loadtxt(filename, skiprows=skiprows)

    return hist, dr