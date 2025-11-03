import os #import os for working directory changes

os.chdir(os.path.dirname(os.path.abspath(__file__))) #change the working directory to the file directory to avoid long file names

from parameters import n_particles, dt, n_steps, n_save

import Initialize as Init

from Plot import set_Plot_Font
from Simulate import simulate


##### units:
# kBT = 1    

set_Plot_Font() #Set global plot font size, to correspond to the latex page size


positions = Init.create_particles(n_particles, n_steps)



positions = simulate(positions,
            n_steps,
            dt,
            n_save,
            save_to_file=True,
            Analyze = False,
             )
