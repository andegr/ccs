import numpy as np
from numba import njit
from parameters import dimensions, L, n_particles, rho

def create_particles(number_of_particles, n_steps, n_steps_equil, n_save, dimensions=dimensions):
    """Creates only the (Particles, Dimensions, Trajectory_steps) array"""
    positions_equil = np.zeros((number_of_particles, dimensions, n_steps_equil // n_save))
    positions = np.zeros((number_of_particles, dimensions, n_steps // n_save))
    
    positions_equil = initialize_part_pos_3D(positions_equil)   # only positions_equil[:,:,0] should be changed

    return positions, positions_equil

def initialize_part_pos_3D(positions_equil):
    """
    Sets particles on a 3D Simple Cubic lattice-like structure based on
    the required number density.
    """
    
    # L = (n_particles / rho)**(1/3)
    
    # Determine the side length of rounded up value of n_particles
    N_side = int(np.ceil(n_particles**(1/3)))
    
    # 3. uniform spacing between particles (lattice constant)
    d_xyz = L / N_side
    
    print(f"Box Length (L): {L:.2f}")
    print(f"Grid Size (N_side): {N_side}")
    print(f"Spacing (d_xyz): {d_xyz:.3f}")

    particle_index = 0
    
    # Iterate through the grid coordinates (x, y, z)
    for i in range(N_side):  # x-dimension
        for j in range(N_side):  # y-dimension
            for k in range(N_side):  # z-dimension
                
                if particle_index == n_particles:
                    # Stop once N particles have been placed
                    return positions_equil

                # Set the x-coordinate (dimension index 0)
                positions_equil[particle_index, 0, 0] = i * d_xyz
                # Set the y-coordinate (dimension index 1)
                positions_equil[particle_index, 1, 0] = j * d_xyz
                # Set the z-coordinate (dimension index 2)
                positions_equil[particle_index, 2, 0] = k * d_xyz
                
                particle_index += 1
                
    return positions_equil


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

