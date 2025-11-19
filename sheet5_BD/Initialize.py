import numpy as np

def create_particles(number_of_particles, n_steps, n_steps_equil, n_save, dimensions=2):
    positions_equil = np.zeros((number_of_particles, dimensions, n_steps // n_save))
    positions = np.zeros((number_of_particles, dimensions, n_steps // n_save))
    
    return positions, positions_equil