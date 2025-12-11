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
    n_particles: int = 49          # Total number of particles

    # MC Simulation Parameters (Constants)
    kB: float = 1.0
    T: float = 1.0
    sigma: float = 1.0
    max_displacement: float = 0.05    # * sigma, with sigma=1

    # LJ Inputs       # Number density: used to calculate L
    r_cut: float = 4        # * sigma, with sigma=1
    L: float = 10                  # * sigma, with sigma=1

    # Time Related Inputs
    n_sweeps: float = 1e6             # number of times to sweep over all particles 
    n_sweeps_eq: float = 1e6          # number of times to sweep over all particles for equilibration
    n_save: float = 10                # Steps between saving 
    n_save_hist: float = 10            # Steps between saving histogram 

    # Histogram/RDF Inputs
    dr: float = 0.05

    # --- Derived Attributes (Calculated in __post_init__) ---
    # These fields must use field(init=False) as they depend on the inputs above.
    eps: float = field(init=False)      # LJ epsilon (eps = kB * T)
    rho: float = field(init=False)      # density which is defined via n_part * sigma / L**2
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
        # System parameters
        self.rho = self.sigma * self.n_particles / self.L**2

        # Convert sweep counts to integers automatically
        self.n_sweeps = int(self.n_sweeps)
        self.n_sweeps_eq = int(self.n_sweeps_eq)
        self.n_save = int(self.n_save)
        self.n_save_hist = int(self.n_save_hist)
        
        # 1. LJ Parameters
        self.eps = self.kB * self.T                                  # eps = 1 * kB * T

        # 2. Box Size Parameters

        # Periodic Boundary Conditions (PBC) definitions
        # self.xlo = -self.L / 2
        # self.xhi = +self.L / 2
        # self.ylo = -self.L / 2
        # self.yhi = +self.L / 2
        # self.zlo = -self.L / 2
        # self.zhi = +self.L / 2
        self.xlo = 0
        self.xhi = +self.L
        self.ylo = 0
        self.yhi = +self.L
        self.zlo = 0
        self.zhi = +self.L

        # 3. Step Counts (must be integers)
        # # np.ceil rounds up values to the next integer
        # self.n_steps = max(1, np.ceil(self.t_sim / self.dt))        # n_steps = int(t_sim / dt)
        # self.n_steps_equil = max(0, np.ceil(self.t_equil / self.dt)) # n_steps_equil = int(t_equil / dt)

        # 4. Histogram Binning (RDF)
        self.r_max = self.r_cut
        self.num_bins = int(np.floor(self.r_max / self.dr))              # num_bins = int(r_max / dr)

