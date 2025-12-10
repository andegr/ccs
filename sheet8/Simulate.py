from numba import njit
import time
import logging
import numpy as np
from SaveToFile import save_trajectory, save_positions_txt, save_hist
from Plot import plot_hist
from Monte_Carlo import MC_Sweep
from parameters import MCSimulationParameters



def calc_shell_volumes(hist):
    shell_volumes = np.zeros_like(hist)

    for i in range(len(shell_volumes)):
        r_lower = i * dr
        r_upper = r_lower + dr
        shell_volumes[i] = (4 / 3) * np.pi * (r_upper**3 - r_lower**3)
    return shell_volumes



def normalize_hist(hist, n_particles, n_steps, n_ana, rho):
    # correct normalization with number of analyis steps
    ana_steps = n_steps // n_ana
    hist = hist / ana_steps # (np.sum(hist)) 
    shell_volumes = calc_shell_volumes(hist)
    g_r = hist/ (n_particles * rho * shell_volumes)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     g_r = np.where(ideal > 0, hist / ideal, 0.0)  # Avoid division by 0
    return g_r

# @njit
def simulation_loop(positions, n_sweeps, num_bins, n_save_hist, n_save,
    dimensions, n_particles, max_displ, L, r_cut, eps, sigma
):  # Can't pass parameter as one instance to numba functions so we have to pass all seperately

    Analyze = True  

    new_positions = positions[:, :, 0]  # first coordinate slice

    # allocate histogram arrays
    # hist_size = np.int32(num_bins)      # numba needs explicit type
    # cumulative_hist = np.zeros(hist_size, dtype=np.int32)
    # total_counts_hist = 0

    hist = np.zeros(num_bins)

    for n in range(1, n_sweeps-1):
        last_positions = new_positions.copy()
        # main MC sweep update
        new_positions = MC_Sweep(
            last_positions,
            n_particles,
            dimensions,
            max_displ,
            L,
            r_cut,
            eps,
            sigma
        )

        # save configurations every n_save
        if n % n_save == 0:
            idx_int = np.int32(n // n_save)
            positions[:, :, idx_int] = new_positions

            # future RDF/MSD processing goes here...
            # update_hist(cumulative_hist, current_distances, dr)
            # total_counts_hist += 1

    return positions, hist


def simulate(positions, positions_equil, parameters: MCSimulationParameters, save_to_file=True, Analyze = False):
    n_sweeps = parameters.n_sweeps 
    n_sweeps_eq = parameters.n_sweeps_eq
    num_bins = parameters.num_bins
    n_save_hist = parameters.n_save_hist 
    n_save=parameters.n_save
    dimensions = parameters.dimensions
    n_particles = parameters.n_particles
    max_displ=  parameters.max_displacement 
    L = parameters.L
    r_cut = parameters.r_cut 
    eps = parameters.eps
    sigma = parameters.sigma


    start_time = time.time()
    logging.info("Starting equilibration...")
    
    positions_equil, _ = simulation_loop(positions_equil, n_sweeps_eq, num_bins, n_save_hist, n_save,
                                         dimensions, n_particles, max_displ, L, r_cut, eps, sigma)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")
    
    
    positions[:,:,0] = positions_equil[:,:,-1]      # make last equil position first sim position

    logging.info("Starting simulation...")
    positions, hist = simulation_loop(positions, n_sweeps, num_bins, n_save_hist, n_save,
                                      dimensions, n_particles, max_displ, L, r_cut, eps, sigma)

    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")

    # hist_normalized = normalize_hist(hist)



    if save_to_file:
        rho = parameters.rho
        save_positions_txt(positions, parameters, f"trajectory_{rho}.txt")
        save_positions_txt(positions_equil, parameters, f"trajectory_eq_{rho}.txt")
        save_trajectory(positions, parameters, f"trajectory_OVITO_eq_{rho}.txt", 1)
        save_trajectory(positions, parameters, f"trajectory_OVITO_{rho}.txt", 1)
        # save_hist(hist_normalized,dr, f"hist_{rho}_{epsilon}.txt")
        logging.info(f"Finished saving trajectory Ovito and with less overhead")

        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

    # plot_hist(f"hist_{rho}_{epsilon}.txt")

    return None

