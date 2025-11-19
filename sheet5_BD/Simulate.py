from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama
from SaveToFile import save_trajectory, save_timesteps_and_observable

@njit
def integration_loop(positions, dt, n_steps, n_save, Analyze):


    new_positions = positions[:,:,0]        # take first positions

    for n in range(1, n_steps):
        last_positions = new_positions
        new_positions = Euler_Maruyama(last_positions, dt, Analyze)

        if n % n_save == 0:
            idx = n // n_save
            positions[:,:, idx] = new_positions
        
        return positions


def simulate(positions, positions_equil, n_steps, dt, n_save, Analyze=False, save_to_file=False):

    start_time = time.time()


    logging.info("Starting equilibration...")
    
    positions_equil = integration_loop(positions_equil, dt, n_steps, n_save, Analyze)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")
    
    
    positions[:,:,0] = positions_equil[:,:,-1]      # make last equil position first sim position

    logging.info("Starting simulation...")
    positions = integration_loop(positions, dt, n_steps, n_save, Analyze)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    if save_to_file:
        save_trajectory(positions, "trajectory.txt", 1)

        # logging.info(f"comparing shapes: timesteps: {timesteps_arr.shape} and {displ_vec[0,0,:].shape}")

        # saves only the displacement vectors in the last timestep
        # particlenumbers = range(len(displ_vec[:,0,-1]))
        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,0,-1], filename="displ_vec_x_axis.txt")
        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    return None

