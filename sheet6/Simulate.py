from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama
from SaveToFile import save_trajectory, save_positions_txt
from Initialize import set_initial_positions_2part

@njit
def integration_loop(positions, dt, n_steps, n_save, Analyze):

    new_positions = positions[:,:,0]        # take first positions

    for n in range(1, n_steps):
        last_positions = new_positions
        new_positions = Euler_Maruyama(last_positions, dt, Analyze)

        if n % n_save == 0:
            idx = n // n_save
            positions[:,:,idx] = new_positions
        
    return positions


def simulate(positions, positions_equil, n_steps, n_steps_equil,
             dt, n_save, Analyze=False, save_to_file=False):

    positions_equil = set_initial_positions_2part(number_of_particles=2, n_steps=n_steps, 
                                                  n_steps_equil=n_steps_equil, n_save=n_save, dimensions=1)

    start_time = time.time()
    logging.info("Starting equilibration...")
    
    positions_equil = integration_loop(positions_equil, dt, n_steps_equil, n_save, Analyze)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")
    
    
    positions[:,:,0] = positions_equil[:,:,-1]      # make last equil position first sim position

    logging.info("Starting simulation...")
    positions = integration_loop(positions, dt, n_steps, n_save, Analyze)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    if save_to_file:
        save_positions_txt(positions, "trajectory.txt")
        save_trajectory(positions, "trajectory_OVITO.txt", 1)
        logging.info(f"Finished saving trajectory Ovito and with less overhead")

        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    return None

