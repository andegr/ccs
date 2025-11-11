dimensions = 1
n_particles = 1 #Total number of particles
dt = 0.1 # timestep
t_sim = 5e5 # simulation time

# setting
# sigma = 1; xi = 1;
kB = 1
T = 1
tau_BD = 1 # sigma**2 * xi / (kB T) = 1
K_H = 10 #* kB * T / sigma**2


friction_coef = 1 # friction coefficient

n_steps = int(t_sim / dt)
n_save = 1000

# # RDF g(r) parameters
# number_of_bins = int(5e2)
# binwidth = L/number_of_bins/2
    