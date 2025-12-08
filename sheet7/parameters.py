dimensions = 3
n_particles = 500 #Total number of particles

# MD Simulation Parameters
# setting
# sigma = 1; xi = 1;
kB = 1
T = 1
tau_BD = 1 # sigma**2 * xi / (kB T) = 1
sigma = 1

friction_coef = 1 # friction coefficient

# LJ Parameters
eps = 10*kB*T    # = 1 (epsilon)
r_cut = 2.5*sigma   # = 2.5

# Box Size Parameters
# Volume = L**3
rho = 0.9 * sigma**3
# rho = n_particles * sigma**3 / L**3
L = (n_particles * sigma**3 / rho)**(1/3)
xlo = 0; xhi = L
xlo = -L/2; xhi = +L/2
ylo = -L/2; yhi = +L/2
zlo = -L/2; zhi = +L/2

# Histogramm binning for RDF
dr = 0.05
r_max = L/2
num_bins = int(r_max / dr)


dt = 1e-4 * tau_BD  # != 1e-4 (timestep size)
t_sim = 10 * tau_BD # simulation time final !>=100
t_equil = 10 * tau_BD  


n_steps = int(t_sim / dt)
n_steps_equil = int(t_equil / dt)
n_save = 10
n_ana = 10
    

# from dataclasses import dataclass, field
# import math


# @dataclass
# class MDSimulationParameters:
#     """
#     A dataclass to hold and derive parameters for a Molecular Dynamics (MD) simulation.

#     Input values can be overridden when creating an instance. Derived values are
#     calculated automatically in __post_init__.
#     """

#     # --- Primary Inputs & Constants (No ClassVar needed for simplicity) ---
#     dimensions: int = 3
#     n_particles: int = 15          # Total number of particles

#     # MD Simulation Parameters (Constants)
#     kB: float = 1.0
#     T: float = 1.0
#     tau_BD: float = 1.0            # Brownian Relaxation Time
#     sigma: float = 1.0
#     friction_coef: float = 1.0     # friction coefficient

#     # Density and LJ Inputs
#     rho: float = 0.5               # Number density: used to calculate L
#     r_cut_factor: float = 2.5      # Factor used to calculate r_cut = r_cut_factor * sigma

#     # Time Inputs
#     dt: float = 0.0001             # Timestep size (1e-4 * tau_BD)
#     t_sim: float = 1000.0          # Simulation time final (1000 * tau_BD)
#     t_equil: float = 10.0          # Equilibration time (10 * tau_BD)
#     n_save: int = 10               # Steps between saving data

#     # Histogram/RDF Inputs
#     dr: float = 0.05

#     # --- Derived Attributes (Calculated in __post_init__) ---
#     # These fields must use field(init=False) as they depend on the inputs above.
#     eps: float = field(init=False)      # LJ epsilon (eps = kB * T)
#     r_cut: float = field(init=False)    # LJ cutoff
#     L: float = field(init=False)        # Box Size
#     r_max: float = field(init=False)    # RDF maximum radius (r_max = r_cut)
#     num_bins: int = field(init=False)   # RDF bins
#     n_steps: int = field(init=False)    # Total steps
#     n_steps_equil: int = field(init=False) # Equilibration steps

#     # Box Coordinates (derived from L)
#     xlo: float = field(init=False)
#     xhi: float = field(init=False)
#     ylo: float = field(init=False)
#     yhi: float = field(init=False)
#     zlo: float = field(init=False)
#     zhi: float = field(init=False)

#     def __post_init__(self):
#         """
#         Calculates all derived parameters based on the primary inputs.
#         """
        
#         # 1. LJ Parameters
#         self.eps = self.kB * self.T                                  # eps = 1 * kB * T
#         self.r_cut = self.r_cut_factor * self.sigma                  # r_cut = 2.5 * sigma

#         # 2. Box Size Parameters
#         # L = (n_particles * sigma^dimensions / rho)^(1/dimensions)
#         L_value = (self.n_particles * self.sigma**self.dimensions / self.rho)**(1/self.dimensions)
#         self.L = L_value

#         # Periodic Boundary Conditions (PBC) definitions
#         self.xlo = -L_value / 2
#         self.xhi = +L_value / 2
#         self.ylo = -L_value / 2
#         self.yhi = +L_value / 2
#         self.zlo = -L_value / 2
#         self.zhi = +L_value / 2

#         # 3. Step Counts (must be integers)
#         self.n_steps = max(1, math.ceil(self.t_sim / self.dt))        # n_steps = int(t_sim / dt)
#         self.n_steps_equil = max(0, math.ceil(self.t_equil / self.dt)) # n_steps_equil = int(t_equil / dt)

#         # 4. Histogram Binning (RDF)
#         self.r_max = self.r_cut
#         self.num_bins = math.floor(self.r_max / self.dr)              # num_bins = int(r_max / dr)

