dimensions = 1
n_particles = 10 #Total number of particles
dt = 0.001 # timestep
t_sim = 2e5 # simulation time

x_R = 1

# setting
# sigma = 1; xi = 1;
kB = 1
T = 1
tau_BD = 1 # sigma**2 * xi / (kB T) = 1


friction_coef = 1 # friction coefficient

n_steps = int(t_sim / dt)
n_save = 100

#------------box set-up-------
x_low = y_low = z_low = 0
x_high = y_high = z_high = 1
    