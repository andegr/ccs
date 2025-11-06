
n_particles = 10 #Total number of particles
dt = 0.0001 # tiemstep
t_sim = 100 # simulation time

# setting
# sigma = 1; xi = 1;
# kB = 1; T = 1
tau_BD = 1 # sigma**2 * xi / (kB T) = 1


friction_coef = 1 # friction coefficient

n_steps = int(t_sim / dt)
n_save = 1000

# # RDF g(r) parameters
# number_of_bins = int(5e2)
# binwidth = L/number_of_bins/2
    