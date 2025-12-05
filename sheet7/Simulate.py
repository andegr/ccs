from numba import njit
import time
import logging
import numpy as np

from IntegrationSchemes import Euler_Maruyama
from SaveToFile import save_trajectory, save_positions_txt, save_hist
from parameters import L, dr, n_ana, eps
from Plot import plot_hist

epsilon = eps



def calc_shell_volumes(hist):
    shell_volumes = np.zeros_like(hist)

    for i in range(len(shell_volumes)):
        r_lower = i * dr
        r_upper = r_lower + dr
        shell_volumes[i] = (4 / 3) * np.pi * (r_upper**3 - r_lower**3)
    return shell_volumes



def normalize_hist(hist):
    hist = hist / (np.sum(hist)) 
    shell_volumes = calc_shell_volumes(hist)
    g_r = hist/shell_volumes
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     g_r = np.where(ideal > 0, hist / ideal, 0.0)  # Avoid division by 0

    return g_r


@njit     # <-- care commenting out
def integration_loop(positions, dt, n_steps, n_save, num_bins, dr, Analyze):

    # NOTE: 'positions' array must store UNWRAPPED coordinates for MSD calculation
    new_positions = positions[:,:,0] # Take first positions
    hist_size = np.int32(num_bins)  # <-- NOTE: not having this crashed via numba !!!
    cumulative_hist = np.zeros(hist_size, dtype=np.int32)
    total_counts_hist = 0

    # n_bins = L/2 / dr
    # n_hist_steps = int(n_steps / n_ana)
    hist = np.zeros(num_bins)

    for n in range(1, n_steps):
        if Analyze and (n_steps%n_ana ==0):
            hist_distances = True
        else:
            hist_distances = False

        last_positions = new_positions
        new_positions, current_distances, hist = Euler_Maruyama(last_positions, dt, hist, hist_distances)


        # NOTE: uncomment the following and comment the above for MSD calculation
        # 1. Get WRAPPED positions for force calculation (using the last saved UNWRAPPED coordinates)
        # wrapped_positions = new_positions % L 
        # 2. Get forces and the new UNWRAPPED positions
        # new_positions, current_distances = Euler_Maruyama(wrapped_positions, dt, L, dimensions, Analyze)

        if n % n_save == 0:
            idx_int = np.int32(n // n_save)
            positions[:,:,idx_int] = new_positions

            #-------cheaper to do it in the force loop-------------
            # # RDF Calculation
            # update_hist(cumulative_hist, current_distances, dr)
            # total_counts_hist += 1
        
    return positions, hist


def simulate(positions, positions_equil, n_steps, n_steps_equil, dt, n_save, num_bins, dr,
              Analyze=False, save_to_file=False):

    start_time = time.time()
    logging.info("Starting equilibration...")
    
    positions_equil, _ = integration_loop(positions_equil, dt, n_steps_equil, n_save, num_bins, dr, Analyze)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")
    
    
    positions[:,:,0] = positions_equil[:,:,-1]      # make last equil position first sim position

    logging.info("Starting simulation...")
    positions, hist = integration_loop(positions, dt, n_steps, n_save, num_bins, dr, True)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    hist_normalized = normalize_hist(hist)



    if save_to_file:
        save_positions_txt(positions, "trajectory_{rho}_{epsilon}.txt")
        # save_positions_txt(positions_equil, "trajectory_eq_{rho}_{epsilon}.txt")
        # save_trajectory(positions, "trajectory_OVITO_eq_{rho}_{epsilon}.txt", 1)
        save_trajectory(positions, "trajectory_OVITO_{rho}_{epsilon}.txt", 1)
        save_hist(hist_normalized,dr, "hist_{rho}_{epsilon}.txt")
        logging.info(f"Finished saving trajectory Ovito and with less overhead")

        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    plot_hist("hist_{rho}_{epsilon}.txt")

    return None

