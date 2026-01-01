from dataclasses import dataclass, field, fields
import numpy as np
from pathlib import Path


@dataclass
class MDSimulationParameters:
    """
    A dataclass to hold and derive parameters for a Monte Carlo (MC) simulation.

    Input values can be overridden when creating an instance. Derived values are
    calculated automatically in __post_init__.
    """

    # --- Primary Inputs & Constants (No ClassVar needed for simplicity) ---
    dimensions: int = 2
    n_particles: int = 500          # Total number of particles

    # MD Simulation Parameters (Constants)
    kB: float = 1.0
    T: float = 1.0
    sigma: float = 1.0
    eps: float = 1.0

    Dr: float = 1.0
    Dt: float = 1.0     # vorerst 0
    F: float = 0 
    v0: float = field(init=False)   # Effective propulsion speed v0 = beta * Dt * F
    

    # LJ Inputs       # Number density: used to calculate L
    r_cut: float = 2.5        # * sigma, with sigma=1
    L: float = 10                  # * sigma, with sigma=1

    # Time Related Inputs
    tau_BD: float = 1
    dt: float = 1e-3        # in units of tau_BD 
    t_sim: float = 100        # in units of tau_BD   
    t_eq: float = 25         # in units of tau_BD
    n_save: int = 10
    
    # --- Derived Attributes (Calculated in __post_init__) ---
            
    n_steps: int = field(init=False)   
    n_steps_saved: int = field(init=False)  # number of saved steps
    n_steps_eq: int = field(init=False)   


    # --- Derived Attributes (Calculated in __post_init__) ---
    # These fields must use field(init=False) as they depend on the inputs above.
    rho: float = field(init=False)      # density which is defined via n_part * sigma / L**2
    # num_bins: int = field(init=False)   # RDF bins
    
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
        # Self propulsion velocity: v0 = beta * Dt * F
        self.v0 = self.Dt * self.F / (self.kB*self.T)

        # Time Parameters
        self.dt = self.dt * self.tau_BD
        self.t_sim: float = self.t_sim * self.tau_BD     
        self.t_eq: float = self.t_eq * self.tau_BD     
        
        self.n_steps = int(self.t_sim / self.dt)
        self.n_steps_saved = int(self.n_steps / self.n_save)

        self.n_steps_eq = int(self.t_eq / self.dt)

        # System parameters
        self.rho = self.sigma * self.n_particles / self.L**2

        # Convert sweep counts to integers automatically
        self.n_save = int(self.n_save)
        
        # 1. LJ Parameters
        self.eps = self.kB * self.T                                  # eps = 1 * kB * T

        # 2. Box Size Parameters
        self.xlo = 0
        self.xhi = +self.L
        self.ylo = 0
        self.yhi = +self.L
        self.zlo = 0
        self.zhi = +self.L

        # 3. Step Counts (must be integers)
        # # np.ceil rounds up values to the next integer
        # self.n_steps = max(1, np.ceil(self.t_sim / self.dt))        # n_steps = int(t_sim / dt)
        # self.n_steps_equil = max(0, np.ceil(self.t_eq / self.dt)) # n_steps_equil = int(t_eq / dt)

