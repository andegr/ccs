dimensions = 1
n_particles = 1 #Total number of particles

# setting
# sigma = 1; xi = 1;
kB = 1
T = 1
tau_BD = 1 # sigma**2 * xi / (kB T) = 1
sigma = 1


dt = 0.001 # timestep
t_sim = 1e5 # simulation time final != 1e7
t_equil = 500   # 500*tau_BD
t_acorr = 10    # 10*tau_BD


friction_coef = 1 # friction coefficient

V_0 = kB*T          # = 1
A = 0.5/sigma**4
B = 2 / sigma**2
C = 0.1 / sigma

n_steps = int(t_sim / dt)
n_steps_equil = int(t_equil / dt)   # = 5e5
n_steps_acorr = int(t_acorr / dt)
n_save = 100
    