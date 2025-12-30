import os #import os for working directory changes
os.chdir(os.path.dirname(os.path.abspath(__file__))) #change the working directory to the file directory to avoid long file names
import Initialize as Init
from Plot import set_Plot_Font
from Simulate import simulate
from utilities import utils
import logging
from parameters import MCSimulationParameters

##### units:
# kBT = 1    

def main(run_task="all"):
    parameters = MCSimulationParameters()     # gets initialized with parameters defined in constructor of Object

    if run_task in ("all", "1"):
        set_Plot_Font() #Set global plot font size, to correspond to the latex page size

        logs_dir = utils.create_logs_directory()
        utils.setup_logging(logs_dir)
        logging.info(f'Created logging directory: {logs_dir}')
        plots_dir = utils.create_plots_directory()
        outputs_dir = utils.create_outputs_directory()

        positions, positions_eq, orientations, orientations_eq = Init.create_particles(parameters) 

        simulate(positions,
                 positions_eq,
                 orientations,
                 orientations_eq,
                 parameters,               # parameters of MCSimulation which can acces all relevant parameters
                 outputs_dir,
                 save_to_file=True,
                 Analyze = False)



# if __name__ == '__main__':
main(run_task="1")