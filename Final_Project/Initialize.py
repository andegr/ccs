import numpy as np
from logging import info
# from numba import njit
from parameters import MDSimulationParameters

def create_particles(parameters: MDSimulationParameters):
    """Creates only the (Particles, Dimensions, Trajectory_steps) array"""
    dim = parameters.dimensions
    n_particles = parameters.n_particles
    n_steps_saved = parameters.n_steps_saved
    n_steps_saved_eq = parameters.n_steps_saved_eq

    # has form = (number particles, position component, orientation angle theta, trajectory step)
    positions_equil = np.zeros((n_particles, dim, n_steps_saved_eq))    # in 3 we store the orientation vector
    positions = np.zeros((n_particles, dim, n_steps_saved))

    thetas_equil = np.zeros((n_particles, n_steps_saved_eq))
    thetas = np.zeros((n_particles, n_steps_saved))

    positions_equil, thetas_equil = initialize_part_pos_2D(positions_equil, thetas_equil, parameters)   

    return positions, positions_equil, thetas, thetas_equil

def initialize_part_pos_2D(positions_equil, thetas_equil, parameters: MDSimulationParameters):
    """
    Sets particles on a 3D Simple Cubic lattice-like structure based on
    the required number density.
    """
    L = parameters.L
    n_particles = parameters.n_particles
    disc = parameters.disc
    r_disc = parameters.r_disc
        
    if disc:
        # -------------------------------------------------
        # Disc confinement: equal-area distribution in 2D
        # -------------------------------------------------

        R = r_disc
        N = n_particles

        # (1) number of radial shells (≈ sqrt(N))
        N_r = int(np.ceil(np.sqrt(N)))  # (1)

        # (2) radial spacing using equal-area condition
        dr = 1  # (2)

        particle_index = 0
        random_thetas = np.random.uniform(0, 2 * np.pi, n_particles)

        for i in range(N_r):
            # (3) radius at shell center (equal-area)
            r = (i + 0.5) * dr  # (3)

            # (4) number of particles on this ring ~ circumference
            n_theta = max(1, int(2 * np.pi * r / dr))  # (4)

            for j in range(n_theta):
                if particle_index == N:
                    return positions_equil, thetas_equil

                # (5) angular position
                phi = 2 * np.pi * j / n_theta  # (5)

                # (6) Cartesian coordinates
                positions_equil[particle_index, 0, 0] = r * np.cos(phi)  + L/2
                positions_equil[particle_index, 1, 0] = r * np.sin(phi)  + L/2

                # random orientation
                thetas_equil[particle_index, 0] = random_thetas[particle_index]

                particle_index += 1


    else:
        # -------------------------------------------------
        # Distribution on lattice
        # -------------------------------------------------


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