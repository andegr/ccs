dimensions = 1
dimensions_task1 = 1
n_particles = 2 #Total number of particles

n_particles_task1 = 2 #Total number of particles

# setting
# sigma = 1; xi = 1;
kB = 1
T = 1
tau_BD = 1 # sigma**2 * xi / (kB T) = 1
sigma = 1

L = 10 * sigma  # = 10
xlo = 0; xhi = L
# xlo = -L/2; xhi = +L/2
# ylo = -L/2; yhi = +L/2
# zlo = -L/2; zhi = +L/2

eps = 1*kB*T    # = 1 (epsilon)

r_cut = 2.5*sigma   # = 2.5


dt = 1e-4 * tau_BD  # != 1e-4 (timestep size)
t_sim = 1000 * tau_BD # simulation time final != 1e7
t_equil = 10 * tau_BD   # 500*tau_BD


friction_coef = 1 # friction coefficient


n_steps = int(t_sim / dt)
n_steps_equil = int(t_equil / dt)
n_save = 10
    