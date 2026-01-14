import os #import os for working directory changes
# os.chdir(os.path.dirname(os.path.abspath(__file__))) #change the working directory to the file directory to avoid long file names
print(os.getcwd())
import Initialize as Init
from Plot import set_Plot_Font
from Simulate import simulate
import utilities as utils
import logging
from parameters import MDSimulationParameters

##### units:
# kBT = 1    

def main(run_task="all"):
    parameters_base = MDSimulationParameters()     # gets initialized with parameters defined in constructor of Object

    if run_task in ("all", "1"):#
        set_Plot_Font() #Set global plot font size, to correspond to the latex page size

        logs_dir = utils.create_logs_directory()
        utils.setup_logging(logs_dir)
        logging.info(f'Created logging directory: {logs_dir}')
        plots_dir = utils.create_plots_directory()
        outputs_dir = utils.create_outputs_directory()


        for run_id in range(parameters_base.multiruns):

            parameters = MDSimulationParameters(run_id=run_id)

            positions, positions_eq, orientations, orientations_eq = Init.create_particles(parameters) 

            simulate(positions,
                    positions_eq,
                    orientations,
                    orientations_eq,
                    parameters,
                    outputs_dir,
                    save_to_file=True,
                    save_to_file_eq=True,
                    Analyze = False,)



# if __name__ == '__main__':
main(run_task="1")