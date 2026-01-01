import numpy as np
# from numba import njit
from parameters import MDSimulationParameters

def create_particles(parameters: MDSimulationParameters):
    """Creates only the (Particles, Dimensions, Trajectory_steps) array"""
    dim = parameters.dimensions
    n_particles = parameters.n_particles
    n_steps = parameters.n_steps
    n_steps_eq = parameters.n_steps_eq
    n_save = parameters.n_save

    # has form = (number particles, position component, orientation angle theta, trajectory step)
    positions_equil = np.zeros((n_particles, dim, n_steps_eq // n_save))    # in 3 we store the orientation vector
    positions = np.zeros((n_particles, dim, n_steps // n_save))

    thetas_equil = np.zeros((n_particles, n_steps_eq // n_save))
    thetas = np.zeros((n_particles, n_steps // n_save))

    positions_equil, thetas_equil = initialize_part_pos_2D(positions_equil, thetas_equil, parameters)   

    return positions, positions_equil, thetas, thetas_equil

def initialize_part_pos_2D(positions_equil, thetas_equil, parameters: MDSimulationParameters):
    """
    Sets particles on a 3D Simple Cubic lattice-like structure based on
    the required number density.
    """
    L = parameters.L
    n_particles = parameters.n_particles
    
    
    # Determine the side length of rounded up value of n_particles
    N_side = int(np.ceil(n_particles**(1/2)))
    
    # 3. uniform spacing between particles (lattice constant)
    d_xy = L / N_side
    
    print(f"Box Length (L): {L:.2f}")
    print(f"Grid Size (N_side): {N_side}")
    print(f"Spacing (d_xy): {d_xy:.3f}")

    particle_index = 0
    random_thetas = np.random.uniform(0, 2 * np.pi, n_particles)
    
    # Iterate through the grid coordinates (x, y, z)
    for i in range(N_side):  # x-dimension
        for j in range(N_side):  # y-dimension
                                
                if particle_index == n_particles:
                    # Stop once N particles have been placed
                    return positions_equil, thetas_equil

                # Set the x-coordinate (dimension index 0)
                positions_equil[particle_index, 0, 0] = i * d_xy
                positions_equil[particle_index, 1, 0] = j * d_xy
                # random theta-Initialisierung
                thetas_equil[particle_index, 0] = random_thetas[particle_index]
                
                particle_index += 1
                
    return positions_equil, thetas_equil


# def InitializeAtoms(random_seed, settings):
#     # np.random.seed(settings.random_seed)    

#     x = np.zeros(shape=(settings.N))
#     y = np.zeros(shape=(settings.N))
#     z = np.zeros(shape=(settings.N))
#     vx = np.zeros(shape=(settings.N))
#     vy = np.zeros(shape=(settings.N))
#     vz = np.zeros(shape=(settings.N))

#     n = 0
#     i = 0
#     while n < settings.N:
#         x0 =  np.random.rand()*settings.L
#         y0 =  np.random.rand()*settings.L
#         z0 =  np.random.rand()*settings.L

#         b = False
#         for i in range(n):
#             rijx = pbc(x[i], x0, 0, settings.L)
#             rijy = pbc(y[i], y0, 0, settings.L)
#             rijz = pbc(z[i], z0, 0, settings.L)
            
#             r2 = rijx * rijx + rijy * rijy + rijz * rijz
#             r = np.sqrt(r2)

#             if r < settings.sig:
#                 b = True #reject placement
            
#         if not b:
#             x[n] = x0
#             y[n] = y0
#             z[n] = z0

#             vx0 = 0.5 - np.random.rand()
#             vy0 = 0.5 - np.random.rand()
#             vz0 = 0.5 - np.random.rand()
            
#             vx[n] = vx0
#             vy[n] = vy0
#             vz[n] = vz0

#             n+= 1
#         else:
#             i +=1
#             pass # tries same n again

#         if i> settings.N**3:
#             print('could not find enough positions, density to high')
#             break

    
#     return x, y, z, vx, vy, vz

