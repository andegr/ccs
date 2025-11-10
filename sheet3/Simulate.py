from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama
from SaveToFile import save_trajectory, save_timesteps_and_observable

@njit
def integration_loop(positions, dt, n_steps, Analyze):
    displ_vec = np.empty_like(positions)
    for n in range(1, n_steps-1):
        # alter Code:
        # positions[:, n+1, :] = Euler_Maruyama(positions[:, n+1, :], dt, Analyze) 
        # hier muss n statt (n+1) in Euler_Maruyama rein:
        # Neuer Code mit neuer Reihenfolge des ndarray, nämlich (number_of_particles, dimensions, n_steps)
        positions[:,:, n+1], displ_vec[:,:,n+1] = Euler_Maruyama(positions[:,:, n], dt, Analyze)

    return displ_vec


def simulate(positions, n_steps, dt, n_save, Analyze=False, save_to_file=False):

    logging.info("Starting simulation...")
    start_time = time.time()



    displ_vec = integration_loop(positions, dt, n_steps, Analyze)
    

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    if save_to_file:
        save_trajectory(positions, "trajectory.txt", n_save)

        timesteps_arr = np.arange(0, n_steps) * dt

        logging.info(f"comparing shapes: timesteps: {timesteps_arr.shape} and {displ_vec[0,0,:].shape}")

        # saves only the first particl's displacement vectors
        save_timesteps_and_observable(timesteps=timesteps_arr, observable=displ_vec[0,0,:], filename="displ_vec_x_axis.txt")
        save_timesteps_and_observable(timesteps=timesteps_arr, observable=displ_vec[0,1,:], filename="displ_vec_y_axis.txt")

    return positions

