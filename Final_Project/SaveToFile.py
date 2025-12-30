import numpy as np
from parameters import MCSimulationParameters
# Assuming MCSimulationParameters is imported from parameters
# from parameters import MCSimulationParameters 
# You will need to define or import MCSimulationParameters here

# Placeholder for MCSimulationParameters class structure for clarity:
# class MCSimulationParameters:
#     dt: float
#     n_save: int
#     dimensions: int
#     xlo: float
#     xhi: float
#     ylo: float
#     yhi: float
#     zlo: float
#     zhi: float


#----------------------------Ovito Trajectory:----------------------------#

def save_trajectory(
    positions: np.ndarray,
    parameters,  # Use a type hint for MCSimulationParameters
    file_name: str = "trajectory.txt",
    save_interval: int = 1
) -> None:
    """
    Saves positions data to a .txt file formatted for Ovito.

    Parameters:
    - positions (np.ndarray): Array of particle positions (N_particles, dimensions, N_timesteps).
    - parameters: An instance of MCSimulationParameters containing simulation metadata.
    - file_name (str): Output file name.
    - save_interval (int): Only saves every 'save_interval' timestep.
    """
    # dt_save = parameters.dt * parameters.n_save
    dt_save = parameters.n_save
    
    # Use parameters from the object
    xlo, xhi = parameters.xlo, parameters.xhi
    ylo, yhi = parameters.ylo, parameters.yhi
    zlo, zhi = parameters.zlo, parameters.zhi
    dimensions = parameters.dimensions
    
    N_particles, D_check, N_timesteps = positions.shape
    
    if D_check != dimensions:
        raise ValueError(f"Dimensions in positions array ({D_check}) do not match parameters ({dimensions}).")
    
    # Open a file and wipe it, then open for appending
    with open(file_name, "w") as file:
        file.write("")
    with open(file_name, "a") as file:
        for t in range(0, N_timesteps, save_interval):
            file.write("ITEM: TIMESTEP\n"
                       f"{t*dt_save}\n")
            file.write("ITEM: NUMBER OF ATOMS\n"
                       f"{N_particles}\n")
            file.write("ITEM: BOX BOUNDS\n"
                       f"{xlo} {xhi} xlo xhi\n"
                       f"{ylo} {yhi} ylo yhi\n"
                       f"{zlo} {zhi} zlo zhi\n")
            file.write("ITEM: ATOMS id type x y z\n")
            for i in range(N_particles):
                # Retrieve coordinates, using 0.0 for missing dimensions
                x = positions[i, 0, t]
                y = positions[i, 1, t] if dimensions > 1 else 0.0
                z = positions[i, 2, t] if dimensions > 2 else 0.0
                file.write(f"{i} {i} {x} {y} {z}\n")
    
    print(f"Trajectory data saved in file: {file_name}")

def load_trajectory(file_name: str, parameters) -> np.ndarray:
    """
    Loads positions data from an Ovito-formatted trajectory file.

    Parameters:
    - file_name (str): Input trajectory file name.
    - parameters: An instance of MCSimulationParameters to get the dimensions.

    Returns:
    - np.ndarray: Array of particle positions (N_particles, dimensions, N_timesteps).
    """
    dimensions = parameters.dimensions
    
    with open(file_name, "r") as file:
        lines = file.readlines()

    timesteps = []
    positions = []

    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            # Read timestep (float)
            timestep = float(lines[i + 1].strip())
            timesteps.append(timestep)

            # Number of atoms line
            num_atoms = int(lines[i + 3].strip())

            # Move index to the first atom line after the "ITEM: ATOMS ..." header
            i += 9

            # Prepare array for current timestep (N_particles × dimensions)
            pos_t = np.zeros((num_atoms, dimensions))

            for j in range(num_atoms):
                parts = lines[i + j].strip().split()
                # Always read x (index 2 in the ATOMS line: id type x y z)
                pos_t[j, 0] = float(parts[2])
                if dimensions > 1:
                    pos_t[j, 1] = float(parts[3])
                if dimensions > 2:
                    pos_t[j, 2] = float(parts[4])

            positions.append(pos_t)
            i += num_atoms
        else:
            i += 1

    # Combine all timesteps into a single array: (N_particles, dimensions, N_timesteps)
    positions = np.stack(positions, axis=2)

    print(f"Loaded trajectory from {file_name}")
    return positions


#----------------------------Less Overhead Trajectory:----------------------------#
def save_positions_txt(positions: np.ndarray, parameters, filename: str) -> None:
    """
    Saves positions of shape (N_particles, dimensions, N_timesteps)
    to a txt file with minimal overhead.

    Parameters:
    - positions (np.ndarray): Array of particle positions.
    - parameters: An instance of MCSimulationParameters to get the dimensions.
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
    
    Note: This function does not require MCSimulationParameters as 
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
    
    Note: This function does not require MCSimulationParameters.

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
    
    Note: This function does not require MCSimulationParameters.

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