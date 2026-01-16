from dataclasses import dataclass, field, fields, replace
import numpy as np
from pathlib import Path
from helpers import fmt_float, build_traj_fname
from logging import info
from itertools import product

# Functions needed for checking set of parameters (see expand method of Parameters class)
def _is_sweep_value(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray)) and not isinstance(x, (str, bytes))

def _has_any_sweep_field(obj) -> bool:
    for f in fields(obj):
        if not f.init:
            continue
        if _is_sweep_value(getattr(obj, f.name)):
            return True
    return False

@dataclass
class MDSimulationParameters:
    """
    A dataclass to hold and derive parameters for a Monte Carlo (MC) simulation.

    Input values can be overridden when creating an instance. Derived values are
    calculated automatically in __post_init__.
    """

    # --- Primary Inputs & Constants (No ClassVar needed for simplicity) ---
    multiruns: int = 1
    run_id: int = 0
    dimensions: int = 2
    n_particles: int = 250          # Total number of particles
    walls: bool = False
    pairwise: bool = True           # pairwise particle interactions on / off
    sssave_ovito_file: bool = False
    save_ovito_file_eq: bool = False
    save_position_file: bool = True
    save_orientation_file: bool = False

    compute_L_from_area_frac = False

    # MD Simulation Parameters (Constants)
    kB: float = 1.0
    T: float = 1.0
    sigma: float = 1.0
    eps: float = 1.0            # in units of kB T

    Dr: float = 1.0
    Dt: float = 10.0     # vorerst 0
    F: float = field(init=False) 
    v0: float = 5  # Effective propulsion speed v0 = beta * Dt * F
    

    # LJ Inputs       # Number density: used to calculate L
    r_cut: float = 2**(1/6)     # * sigma, with sigma=1, due to WCA
    r_cut_clus: float = 1.5 * r_cut   # arbitrary for now
    L: float = 20                  # units of sigma. CAN BE OVERWRITTEN BY AREA FRACTION
    area_fraction: float = 0.3


    # Time Related Inputs
    tau_BD: float = 1
    dt: float = 1e-3        # in units of tau_BD 
    t_sim: float = 600       # in units of tau_BD   
    t_eq: float = 150         # in units of tau_BD
    n_save: int = 1000
    
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

    # Filenames
    fname_pos: str = field(init=False)
    fname_ori: str = field(init=False)
    fname_OVITO: str =field(init=False)
    fname_pos_eq: str = field(init=False)
    fname_ori_eq: str = field(init=False)
    fname_OVITO_eq: str =field(init=False)
    
    
    def __post_init__(self):
        """
        Calculates all derived parameters based on the primary inputs.
        """
        # If this is a sweep-template object (any parameter is an array/list),
        # don't compute derived scalar stuff like L, filenames, etc.
        if _has_any_sweep_field(self):
            return

        # Self propulsion velocity: v0 = beta * Dt * F
        self.F = self.v0 * self.kB*self.T / self.Dt

        # Time Parameters
        self.dt = self.dt * self.tau_BD
        self.t_sim: float = self.t_sim * self.tau_BD     
        self.t_eq: float = self.t_eq * self.tau_BD     
        
        # Convert sweep counts to integers automatically
        self.n_save = int(self.n_save)

        self.n_steps = int(self.t_sim / self.dt)
        self.n_steps_eq = int(self.t_eq / self.dt)

        self.n_steps_saved = int( (self.n_steps-1)//self.n_save + 1) # needed ?
        self.n_steps_saved_eq = int( (self.n_steps_eq-1)//self.n_save + 1)


        # System parameters
        if self.compute_L_from_area_frac:
            self.L = np.sqrt( self.n_particles * np.pi * self.sigma**2 / (4*self.area_fraction) )
            info(f"Computed L from area fraction eta={self.area_fraction} !!! (not the one set in parameters)")

        self.rho = self.sigma * self.n_particles / self.L**2

        
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


        # Filenames (build once, derive others)
        self.fname_pos = build_traj_fname(
            "traj_positions",
            self.n_particles,
            self.t_sim,
            self.t_eq,
            self.dt,
            self.L,
            self.v0,
            self.Dt,
            self.Dr,
            self.run_id,
            walls=self.walls,
            pairwise=self.pairwise,
            eta=self.area_fraction,
        )
        
        self.fname_ori   = self.fname_pos.replace("traj_positions", "traj_orientations", 1)
        self.fname_OVITO = self.fname_pos.replace("traj_positions", "traj_OVITO", 1).replace(".txt", ".dump", 1)
        self.fname_pos_eq = self.fname_pos.replace("traj_positions", "traj_positions_eq", 1)
        self.fname_ori_eq = self.fname_pos.replace("traj_positions", "traj_orientations", 1)
        self.fname_OVITO_eq = self.fname_pos.replace("traj_positions", "traj_OVITO_eq", 1).replace(".txt", ".dump", 1)


    # ----- Function used to run varying parameter sets like v0 = [1, 2, 4, 8, 16] ------- #
    def expand(self):
        sweep_names, sweep_lists = [], []
        fixed_kwargs = {}

        for f in fields(self):
            if not f.init:
                continue  # <-- IMPORTANT
            val = getattr(self, f.name)
            if _is_sweep_value(val):
                sweep_names.append(f.name)
                sweep_lists.append(list(val))
            else:
                fixed_kwargs[f.name] = val

        if not sweep_names:
            return [self]

        out = []
        for combo in product(*sweep_lists):
            kwargs = dict(fixed_kwargs)
            for name, v in zip(sweep_names, combo):
                kwargs[name] = v
            out.append(replace(self, **kwargs))
        return out