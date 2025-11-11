import numpy as np
from parameters import dt, n_save, dimensions


# Save To File:
# - Saves data from the "data_array" to a .txt file specified in "file_name" parameter
# - The data is formatted to be used by Ovito

#----------------------------Ovito Trajectory:----------------------------#

def save_trajectory(positions, file_name = "trajectory.txt",save_interval = 1):
    dt_save = dt * n_save
    #set min and max bounds of box to the calculated values to get cubic box:
    x_low = y_low = z_low = 0
    x_high = y_high = z_high = 0
    
    N_particles, dimensions, N_timesteps = positions.shape
    
    # Open a file with the given filename. To avoid appending an already existing file, the file is first wiped completely:
    with open(file_name, "w") as file:
        file.write("")
    with open(file_name, "a") as file:
        for t in range(0,N_timesteps,save_interval):
            file.write("ITEM: TIMESTEP\n"
                       f"{t*dt_save}\n")
            file.write("ITEM: NUMBER OF ATOMS\n"
                       f"{N_particles}\n")
            file.write("ITEM: BOX BOUNDS\n"
                       f"{x_low} {x_high} xlo xhi\n"
                       f"{y_low} {y_high} ylo yhi\n"
                       f"{z_low} {z_high} zlo zhi\n")
            file.write("ITEM: ATOMS id type x y z\n")
            for i in range(N_particles):
                x = positions[i, 0, t]
                y = positions[i, 1, t] if dimensions > 1 else 0.0
                z = positions[i, 2, t] if dimensions > 2 else 0.0
                file.write(f"{i} {i} {x} {y} {z}\n")
    
    print(f"Trajectory data saved in file: {file_name}")
    return

def load_trajectory(file_name="trajectory.txt", dimensions=dimensions):
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
                # Always read x
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

#----------------------------Save Observables----------------------------#

def save_timesteps_and_observable(timesteps: np.ndarray, observable: np.ndarray, filename: str):
    """
    Writes timesteps and corresponding energies to a text file with two columns.

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

    Parameters:
    - filename (str): Input text file name.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Two arrays, (timesteps: int array, observable: float array).
    """
    data = np.loadtxt(filename, skiprows=1)  
    timesteps = data[:, 0].astype(int)      
    observable = data[:, 1]                
    return timesteps, observable       


