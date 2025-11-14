from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama
from SaveToFile import save_trajectory, save_timesteps_and_observable, save_observable
from parameters import t_sim, n_particles

@njit
def integration_loop(positions, dt, n_steps, n_save, Analyze):
    displ_vec = np.empty_like(positions)

    new_positions = positions[:,:,0]
    absorption_counter = 0
    last_passage_timestamps = np.zeros(n_particles)
    fp_times = np.zeros(int(50*t_sim)) # estimation for an upper limit - be careful!

    for n in range(1, n_steps):
        t = n * dt
        last_positions = new_positions
        new_positions, new_displacements, absorption_counter, last_passage_timestamps, fp_times = Euler_Maruyama(
            last_positions,
            t,
            absorption_counter,
            last_passage_timestamps,
            fp_times,
            dt,
            Analyze)

        if n % n_save == 0:
            idx = n // n_save
            positions[:,:, idx], displ_vec[:,:,idx] = new_positions, new_displacements


        fp_times_clean = fp_times[:absorption_counter]
    
    return positions, absorption_counter, fp_times_clean


def simulate(positions, n_steps, dt, n_save, Analyze=False, save_to_file=False):

    logging.info("Starting simulation...")
    start_time = time.time()

    positions, absorption_counter, fp_times = integration_loop(positions, dt, n_steps, n_save, Analyze)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")
    logging.info(f"Number of absorptions: {absorption_counter}")
    logging.info(f"Reaction rate: {absorption_counter/t_sim:.2f}")

    if save_to_file:
        save_trajectory(positions, "trajectory.txt", 1)
        save_observable(fp_times, "fpt.txt")

        # logging.info(f"comparing shapes: timesteps: {timesteps_arr.shape} and {displ_vec[0,0,:].shape}")

        # saves only the displacement vectors in the last timestep
        # particlenumbers = range(len(displ_vec[:,0,-1]))
        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,0,-1], filename="displ_vec_x_axis.txt")
        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    return positions

