import os #import os for working directory changes
# os.chdir(os.path.dirname(os.path.abspath(__file__))) #change the working directory to the file directory to avoid long file names
print(os.getcwd())
import numpy as np
import Initialize as Init
from Plot import set_Plot_Font
from Simulate import simulate
import utilities as utils
import logging
from parameters import MDSimulationParameters

##### units:
# kBT = 1    

def main(run_task="all"):

    if run_task in ("all", "1"):#
        set_Plot_Font() #Set global plot font size, to correspond to the latex page size

        logs_dir = utils.create_logs_directory()
        utils.setup_logging(logs_dir)
        logging.info(f'Created logging directory: {logs_dir}')
        plots_dir = utils.create_plots_directory()
        outputs_dir = utils.create_outputs_directory()

        # params = MDSimulationParameters()     # use for no parameter sweep

        # params = MDSimulationParameters(                # use for parameter sweep
        #     area_fraction=np.array([0.3, 0.3 + 1/30, 0.3 + 2/30, 0.4, 0.4+1/30, 0.4+2/30, 0.5]),
        #     multiruns=1)
        
        params = MDSimulationParameters(                # use for parameter sweep
            v0=[20, 15, 10, 5, 0],
            multiruns=1)

        for run_id in range(params.multiruns):
            for param in params.expand():           # expand() used to get each param set for the sweep of runs
                param.run_id = run_id
                positions, positions_eq, orientations, orientations_eq = Init.create_particles(param)
                

                simulate(positions,
                        positions_eq,
                        orientations,
                        orientations_eq,
                        param,
                        outputs_dir,
                        Analyze = False)



# if __name__ == '__main__':
main(run_task="1")