from numba import njit
import time
import logging

from IntegrationSchemes import Euler_Maruyama
from SaveToFile import save_trajectory

@njit
def integration_loop(positions, dt, n_steps, Analyze):
    for n in range(1, n_steps-1):
        positions[:, n+1, :] = Euler_Maruyama(positions[:, n+1, :], dt, Analyze)

def simulate(positions, n_steps, dt, n_save, Analyze=False, save_to_file=False):

    logging.info("Starting simulation...")
    start_time = time.time()



    integration_loop(positions, dt, n_steps, Analyze)
    

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    if save_to_file:
        save_trajectory(positions, "trajectory.txt", n_save)

    return positions

