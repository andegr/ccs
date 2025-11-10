import numpy as np

def create_particles(number_of_particles, n_steps, dimensions=2):
    positions = np.zeros((number_of_particles, dimensions, n_steps)) # changed the order here
    
    return positions