from numba import njit
import time
import logging
import numpy as np
from SaveToFile import save_trajectory, save_positions_txt, save_hist
from Plot import plot_hist
from Monte_Carlo import MC_Move, update_histogram_all_pairs
from parameters import MCSimulationParameters

parameters = MCSimulationParameters()


max_displ = parameters.max_displacement 
L = parameters.L
rho = parameters.rho


plot_hist(f"sheet8/outputs/hist_rho{rho}_maxDispl{max_displ}.txt")



