from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama, Ornstein_Uhlenbeck
from SaveToFile import save_trajectory, save_timesteps_and_observable

@njit
def integration_loop(positions, positions_OU, dt, n_steps, n_save, Analyze):
    displ_vec = np.empty_like(positions)

    new_positions = positions[:,:,0]
    new_positions_OU = positions_OU[:,:,0]

    for n in range(1, n_steps):
        last_positions = new_positions
        last_positions_OU = new_positions_OU
        new_positions, new_displacements = Euler_Maruyama(last_positions, dt, Analyze)
        new_positions_OU = Ornstein_Uhlenbeck(last_positions_OU, dt, Analyze)

        if n % n_save == 0:
            idx = n // n_save
            positions[:,:, idx], displ_vec[:,:,idx] = new_positions, new_displacements
            positions_OU[:,:,idx] = new_positions_OU
    
    return positions, positions_OU


def simulate(positions, positions_OU, n_steps, dt, n_save, Analyze=False, save_to_file=False):

    logging.info("Starting simulation...")
    start_time = time.time()

    positions, positions_OU = integration_loop(positions, positions_OU, dt, n_steps, n_save, Analyze)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    if save_to_file:
        save_trajectory(positions, "trajectory.txt", 1)
        save_trajectory(positions_OU, "trajectory_OU.txt", 1)

        # logging.info(f"comparing shapes: timesteps: {timesteps_arr.shape} and {displ_vec[0,0,:].shape}")

        # saves only the displacement vectors in the last timestep
        # particlenumbers = range(len(displ_vec[:,0,-1]))
        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,0,-1], filename="displ_vec_x_axis.txt")
        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    return positions

