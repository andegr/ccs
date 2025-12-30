from numba import njit
import time
import logging
import numpy as np
from SaveToFile import save_trajectory, save_positions_txt, save_hist
from Plot import plot_hist
from IntegrationSchemes import Euler_Maruyama
from parameters import MCSimulationParameters


# 2D shell areas needed here not 3D volumes !!!
def calc_shell_areas_2D(hist, dr):
    shell_volumes = np.zeros_like(hist)

    for i in range(len(shell_volumes)):
        r_lower = i * dr
        r_upper = r_lower + dr
        shell_volumes[i] =  np.pi * ((r_upper)**2 - r_lower**2)
    return shell_volumes


# def normalize_hist(hist, hist_counter, dr, n_particles, rho):
#     shell_volumes = calc_shell_areas_2D(hist, dr)
#     g_r = hist / (hist_counter * n_particles * rho * shell_volumes) 
#     return g_r

def normalize_hist(hist, hist_counter, dr, n_particles, rho):
    g_r = np.zeros_like(hist)
    for i in range(len(hist)):
        r = i * dr
        shell_area = np.pi * ((r + dr)**2 - r**2)
        ideal_pairs_in_shell = shell_area * rho * n_particles
        g_r[i] = hist[i] / (hist_counter * ideal_pairs_in_shell)
    return g_r


# @njit
def simulation_loop(positions, orientations, n_steps, num_bins, dr, n_save_hist, n_save,
    dimensions, n_particles, max_displ, L, r_cut, eps, sigma):
    # Can't pass parameter as one instance to numba functions so we have to pass all seperately

    Analyze = True  

    new_positions = positions[:, :, 0]  # first coordinate slice
    new_orientations = orientations[:, :, 0]
    # allocate histogram arrays
    # hist_size = np.int32(num_bins)      # numba needs explicit type
    # cumulative_hist = np.zeros(hist_size, dtype=np.int32)
    # total_counts_hist = 0

    hist = np.zeros(num_bins)
    hist_counter = 0
    acceptance_counter = 0

    # distances_matr = np.zeros((n_particles, n_particles))

    for n in range(n_steps):
        last_positions = new_positions          # eigentlich nicht nötig, nur übersichtlicher
        last_orientations = new_orientations    # same here
        # main MC sweep update
        new_positions, new_orientations = Euler_Maruyama(
            last_positions,
            last_orientations,
            dr,
            dimensions,
            L,
            r_cut,
            eps,
            sigma
        )

        # save configurations every n_save
        if n % n_save == 0:
            idx_int = np.int32(n // n_save)
            positions[:, :, idx_int] = new_positions
            orientations[:,:, idx_int] = new_orientations

            # future RDF/MSD processing goes here...
            # update_hist(cumulative_hist, current_distances, dr)
            # total_counts_hist += 1
        
        # if n % n_save_hist == 0:
        #     hist = update_histogram_all_pairs(new_positions, hist, dr, L)
        #     hist_counter += 1
        
    
    return positions, orientations #     hist, hist_counter


def simulate(positions, positions_eq, orientations, orientations_eq,
             parameters: MCSimulationParameters,
             outputs_dir, save_to_file=True, Analyze = False):
    n_steps = parameters.n_steps 
    n_steps_eq = parameters.n_steps_eq
    num_bins = parameters.num_bins
    dr = parameters.dr
    n_save_hist = parameters.n_save_hist 
    n_save=parameters.n_save
    dimensions = parameters.dimensions
    n_particles = parameters.n_particles
    max_displ = parameters.max_displacement 
    L = parameters.L
    rho = parameters.rho
    r_cut = parameters.r_cut 
    eps = parameters.eps
    sigma = parameters.sigma


    start_time = time.time()
    logging.info("Starting equilibration...")
    
    positions_eq, orientations_eq = simulation_loop(positions_eq, orientations_eq, n_steps_eq, num_bins, dr, n_save_hist, n_save,
                                         dimensions, n_particles, max_displ, L, r_cut, eps, sigma)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")
    
    
    positions[:,:,0] = positions_eq[:,:,-1]      # make last equil position first sim position
    orientations[:,0] = orientations_eq[:,-1]

    logging.info("Starting simulation...")
    positions, orientations = simulation_loop(positions, orientations, n_steps, num_bins, dr, n_save_hist, n_save,
                                      dimensions, n_particles, max_displ, L, r_cut, eps, sigma)
    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    # hist_normalized = normalize_hist(hist, hist_counter, dr, n_particles, rho)

    if save_to_file:
        rho = parameters.rho
        # save_positions_txt(positions, parameters, f"trajectory_{rho}.txt")
        # save_positions_txt(positions_equil, parameters, f"trajectory_eq_{rho}.txt")
        save_trajectory(positions_eq, parameters, outputs_dir / f"trajectory_OVITO_eq_rho{rho}_maxDispl{max_displ}.txt", 1)
        save_trajectory(positions, parameters, outputs_dir / f"trajectory_OVITO_rho{rho}_maxDispl{max_displ}.txt", 1)
        logging.info(f"Finished saving trajectory Ovito")
        # save_hist(hist_normalized, dr, outputs_dir / f"hist_rho{rho}_maxDispl{max_displ}.txt")

        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    # plot_hist(f"hist_{rho}.txt")

    return None

