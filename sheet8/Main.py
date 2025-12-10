import os #import os for working directory changes
os.chdir(os.path.dirname(os.path.abspath(__file__))) #change the working directory to the file directory to avoid long file names
from parameters import n_particles, dt, n_steps, n_steps_equil, n_save, dimensions, num_bins, dr
import Initialize as Init
from Plot import set_Plot_Font
from Simulate import simulate
from utilities import utils
import logging
from parameters import MCSimulationParameters

##### units:
# kBT = 1    

def main(run_task="all"):
    prms = MCSimulationParameters()     # gets initialized with parameters defined in constructor of Object

    if run_task in ("all", "1"):
        set_Plot_Font() #Set global plot font size, to correspond to the latex page size

        output_dir = utils.create_output_directory()
        utils.setup_logging(output_dir)
        logging.info(f'Created output directory: {output_dir}')
        utils.create_plots_directory()

        positions, positions_equil = Init.create_particles(n_particles, n_steps, n_steps_equil, n_save, dimensions=dimensions) 


        simulate(positions,
                 positions_equil,
                 prms,               # parameters of MCSimulation which cann acces all relevant parameters
                 save_to_file=True,
                 Analyze = False)



# if __name__ == '__main__':
main(run_task="1")