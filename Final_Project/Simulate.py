from numba import njit
import time
import gc           # for freeing diskspace
import logging
import numpy as np
from SaveToFile import save_OVITO, save_positions_txt, save_orientations_txt, save_timesteps_and_observable
from Plot import plot_hist
from IntegrationSchemes import Euler_Maruyama
from parameters import MDSimulationParameters


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


@njit
def simulation_loop(positions, orientations, n_steps, n_save,
    dimensions, n_particles, L, r_cut, eps, sigma, dt, kB, T, Dt, Dr, v0):
    # Can't pass parameter as one instance to numba functions so we have to pass all seperately

    Analyze = True  
    new_positions = positions[:, :, 0]  # first coordinate slice
    new_orientations = orientations[:, 0]


    # Save initial state once as frame 0
    positions[:, :, 0] = new_positions
    orientations[:, 0] = new_orientations

    # distances_matr = np.zeros((n_particles, n_particles))

    for n in range(1, n_steps):
        last_positions = new_positions          # eigentlich nicht nötig, nur übersichtlicher
        last_orientations = new_orientations    # same here
        # main MC sweep update
        new_positions, new_orientations = Euler_Maruyama(last_positions, last_orientations,
                                                         dimensions, L, r_cut, eps, sigma,
                                                         dt, kB, T, Dt, Dr, v0)

        # save configurations every n_save
        if n % n_save == 0:
            idx_int = np.int32(n // n_save)
            positions[:, :, idx_int] = new_positions
            orientations[:, idx_int] = new_orientations

            # future RDF/MSD processing goes here...
            # update_hist(cumulative_hist, current_distances, dr)
            # total_counts_hist += 1
        
        # if n % n_save_hist == 0:
        #     hist = update_histogram_all_pairs(new_positions, hist, dr, L)
        #     hist_counter += 1
        
    
    return positions, orientations #     hist, hist_counter


def simulate(positions, positions_eq, orientations, orientations_eq,
             parameters: MDSimulationParameters,
             outputs_dir, save_to_file=True, save_to_file_eq=False, Analyze = False):
    
    n_steps = parameters.n_steps 
    n_steps_eq = parameters.n_steps_eq
    n_save=parameters.n_save
    dimensions = parameters.dimensions
    n_particles = parameters.n_particles
    dt = parameters.dt
    kB = parameters.kB
    T = parameters.T
    L = parameters.L
    rho = parameters.rho
    r_cut = parameters.r_cut 
    eps = parameters.eps
    sigma = parameters.sigma
    v0 = parameters.v0
    Dt = parameters.Dt
    Dr = parameters.Dr
    tsim = parameters.t_sim
    fname_pos = parameters.fname_pos
    fname_ori = parameters.fname_ori
    fname_ori_eq = parameters.fname_ori_eq
    fname_OVITO = parameters.fname_OVITO
    fname_OVITO_eq = parameters.fname_OVITO_eq

    start_time = time.time()
    logging.info("Starting equilibration...")
    
    positions_eq, orientations_eq = simulation_loop(positions_eq, orientations_eq, n_steps_eq, n_save,
                                                    dimensions, n_particles, L, r_cut, eps, sigma, dt,
                                                    kB, T, Dt, Dr, v0)
    logging.info(f"Finished equilibration with a time of {time.time() - start_time:.2f} s")

    if save_to_file_eq:
        logging.info("Saving equilibration data...")
        save_orientations_txt(orientations_eq, outputs_dir / fname_ori_eq)        # saves cos(theta) and sin(theta), NOT thetas !
        save_OVITO(positions_eq, orientations_eq, parameters, outputs_dir / fname_OVITO_eq, 1)
        logging.info("Finished saving equilibration data.")

    
    positions[:,:,0] = positions_eq[:,:,-1]      # make last equil position first sim position
    orientations[:,0] = orientations_eq[:,-1]

    gc.collect()
    del positions_eq
    del orientations_eq


    logging.info("Starting simulation...")
    positions, orientations = simulation_loop(positions, orientations, n_steps, n_save,
                                              dimensions, n_particles, L, r_cut, eps, sigma, dt,
                                              kB, T, Dt, Dr, v0)
    logging.info(f"Finished simulation with a total time of {time.time() - start_time:.2f} s")


    if save_to_file:

        logging.info("Saving Simulation data...")
        # save_positions_txt(positions, parameters, f"trajectory_{rho}.txt")
        # save_positions_txt(positions_equil, parameters, f"trajectory_eq_{rho}.txt")
        # save_orientations_txt(orientations, outputs_dir / fname_ori) 
        save_positions_txt(positions, parameters, outputs_dir / fname_pos)
        logging.info(f"Finished saving trajectory")
        # save_OVITO(positions, orientations, parameters, outputs_dir / fname_OVITO, 1)
        # logging.info(f"Finished saving trajectory Ovito")
        # save_hist(hist_normalized, dr, outputs_dir / f"hist_rho{rho}_maxDispl{max_displ}.txt")

        # save_timesteps_and_observable(timesteps=particlenumbers, observable=displ_vec[:,1,-1], filename="displ_vec_y_axis.txt")

        logging.info("Finished saving Simulation data.")

    return None

