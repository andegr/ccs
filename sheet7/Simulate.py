from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama, update_hist
from SaveToFile import save_trajectory, save_positions_txt

@njit     # <-- care commenting out
def integration_loop(positions, dt, n_steps, n_save, num_bins, dr, Analyze):

    # NOTE: 'positions' array must store UNWRAPPED coordinates for MSD calculation
    new_positions = positions[:,:,0] # Take first positions
    hist_size = np.int32(num_bins)  # <-- NOTE: not having this crashed via numba !!!
    cumulative_hist = np.zeros(hist_size, dtype=np.int32)
    total_counts_hist = 0

    for n in range(1, n_steps):
        last_positions = new_positions
        new_positions, current_distances = Euler_Maruyama(last_positions, dt, Analyze)
        # NOTE: uncomment the following and comment the above for MSD calculation
        # 1. Get WRAPPED positions for force calculation (using the last saved UNWRAPPED coordinates)
        # wrapped_positions = new_positions % L 
        # 2. Get forces and the new UNWRAPPED positions
        # new_positions, current_distances = Euler_Maruyama(wrapped_positions, dt, L, dimensions, Analyze)

        if n % n_save == 0:
            idx_int = np.int32(n // n_save)
            positions[:,:,idx_int] = new_positions

            # RDF Calculation
            update_hist(cumulative_hist, current_distances, dr)
            total_counts_hist += 1
        
    return positions, cumulative_hist, total_counts_hist


def simulate(positions, positions_equil, n_steps, n_steps_equil, dt, n_save, num_bins, dr,
              Analyze=False, save_to_file=False):

    start_time = time.time()
    logging.info("Starting equilibration...")
    
    positions_equil, _, _ = integration_loop(positions_equil, dt, n_steps_equil, n_save, num_bins, dr, Analyze)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")
    
    
    positions[:,:,0] = positions_equil[:,:,-1]      # make last equil position first sim position

    logging.info("Starting simulation...")
    positions, cumulative_hist, total_counts_hist = integration_loop(positions, dt, n_steps, n_save, num_bins, dr, Analyze)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    if save_to_file:
        save_positions_txt(positions, "trajectory.txt")
        save_positions_txt(positions_equil, "trajectory_eq.txt")
        save_trajectory(positions, "trajectory_OVITO_eq.txt", 1)
        save_trajectory(positions, "trajectory_OVITO.txt", 1)
        logging.info(f"Finished saving trajectory Ovito and with less overhead")

        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    return None

