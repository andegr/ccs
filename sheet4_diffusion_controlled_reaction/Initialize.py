import numpy as np

def create_particles(number_of_particles, n_steps, n_save, dimensions=2):
    positions = np.zeros((number_of_particles, dimensions, n_steps // n_save))
    # positions = np.zeros((number_of_particles, dimensions, n_steps))        # testing, did not work
    
    return positions