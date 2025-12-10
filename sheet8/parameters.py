from dataclasses import dataclass, field, fields
import numpy as np



@dataclass
class MCSimulationParameters:
    """
    A dataclass to hold and derive parameters for a Monte Carlo (MC) simulation.

    Input values can be overridden when creating an instance. Derived values are
    calculated automatically in __post_init__.
    """

    # --- Primary Inputs & Constants (No ClassVar needed for simplicity) ---
    dimensions: int = 2
    n_particles: int = 9          # Total number of particles

    # MC Simulation Parameters (Constants)
    kB: float = 1.0
    T: float = 1.0
    sigma: float = 1.0
    max_displacement: float = 1    # set this as big as sigma which sounds reasonable at first glance

    # Density and LJ Inputs
    rho: float = 0.5               # Number density: used to calculate L
    r_cut_factor: float = 2.5      # Factor used to calculate r_cut = r_cut_factor * sigma

    # Time Related Inputs
    n_sweeps: int = 1e6             # number of times to sweep over all particles 
    n_sweeps_eq: int = 1e5          # number of times to sweep over all particles for equilibration
    n_save: int = 10               # Steps between saving data

    # Histogram/RDF Inputs
    dr: float = 0.05

    # --- Derived Attributes (Calculated in __post_init__) ---
    # These fields must use field(init=False) as they depend on the inputs above.
    eps: float = field(init=False)      # LJ epsilon (eps = kB * T)
    r_cut: float = field(init=False)    # LJ cutoff
    L: float = field(init=False)        # Box Size
    r_max: float = field(init=False)    # RDF maximum radius (r_max = r_cut)
    num_bins: int = field(init=False)   # RDF bins

    # Box Coordinates (derived from L)
    xlo: float = field(init=False)
    xhi: float = field(init=False)
    ylo: float = field(init=False)
    yhi: float = field(init=False)
    zlo: float = field(init=False)
    zhi: float = field(init=False)

    def __post_init__(self):
        """
        Calculates all derived parameters based on the primary inputs.
        """
        
        # 1. LJ Parameters
        self.eps = self.kB * self.T                                  # eps = 1 * kB * T
        self.r_cut = self.r_cut_factor * self.sigma                  # r_cut = 2.5 * sigma

        # 2. Box Size Parameters
        # L = (n_particles * sigma^dimensions / rho)^(1/dimensions)
        L_value = (self.n_particles * self.sigma**self.dimensions / self.rho)**(1/self.dimensions)
        self.L = L_value

        # Periodic Boundary Conditions (PBC) definitions
        self.xlo = -L_value / 2
        self.xhi = +L_value / 2
        self.ylo = -L_value / 2
        self.yhi = +L_value / 2
        self.zlo = -L_value / 2
        self.zhi = +L_value / 2

        # 3. Step Counts (must be integers)
        # np.ceil rounds up values to the next integer
        self.n_steps = max(1, np.ceil(self.t_sim / self.dt))        # n_steps = int(t_sim / dt)
        self.n_steps_equil = max(0, np.ceil(self.t_equil / self.dt)) # n_steps_equil = int(t_equil / dt)

        # 4. Histogram Binning (RDF)
        self.r_max = self.r_cut
        self.num_bins = np.floor(self.r_max / self.dr)              # num_bins = int(r_max / dr)

