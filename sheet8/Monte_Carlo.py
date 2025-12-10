import numpy as np
from numba import njit, prange, int32, float64
from parameters import MCSimulationParameters


""" Numba can not handle Object like MCSimulationParameters so we can not pass them as arguments :( """
@njit
def MC_Sweep(positions, n_part, dimensions, max_displ, L, r_cut, eps, sigma):
    for _ in range(n_part):
        # i) randomly pick a particle
        idx = pick_random_particle_index(n_part) # FIXED Error 2
        
        # 1. Store the original position
        x_old = positions[idx, 0]
        y_old = positions[idx, 1]

        # 2. Calculate initial energy contribution
        E_pot_0 = E_potential(positions, idx, L, r_cut, eps, sigma)

        # 3. Calculate proposed new position
        dx, dy = get_random_displacement(max_displ)
        x_proposed = x_old + dx
        y_proposed = y_old + dy

        # 4. Apply Periodic Boundary Condition (PBC) wrapping
        x_proposed = x_proposed - L * np.floor(x_proposed / L) 
        y_proposed = y_proposed - L * np.floor(y_proposed / L)

        # Temporarily apply the move to the array for E_pot_1 calculation
        positions[idx, 0] = x_proposed
        positions[idx, 1] = y_proposed
        
        # ii) Calculate new energy contribution
        E_pot_1 = E_potential(positions, idx, L, r_cut, eps, sigma)
        
        # iii) Calculate Metropolis acceptance rate
        P = np.exp(-(E_pot_1 - E_pot_0))

        # iv) Check for acceptance criterion
        q = np.random.uniform(0, 1)
        
        if q < P:
            # Move accepted: positions are already in the new state.
            pass
        else:
            # Move rejected: Revert positions to the old state (FIXED Error 1)
            positions[idx, 0] = x_old
            positions[idx, 1] = y_old
            
    return positions

@njit
def get_random_displacement(max_displ):
    # Generates a random value in the range [-max_displ, +max_displ] 
    dx = np.random.uniform(-max_displ, max_displ)
    dy = np.random.uniform(-max_displ, max_displ)
    return dx, dy

@njit
def pick_random_particle_index(n_part):
    # Generates a random integer from 0 up to (but not including) n_part
    random_index = np.random.randint(0, n_part)
    return random_index

@njit(parallel=True)
def E_potential(positions, index, L, r_cut, eps, sigma):
    n_particles = positions.shape[0]
    E_pot = 0.0

    for j in prange(n_particles):
        if index != j:
            rijx = pbc_distance(positions[index,0], positions[j,0], 0, L)
            rijy = pbc_distance(positions[index,1], positions[j,1], 0, L)
            
            r2 = rijx**2 + rijy**2
            
            if r2 < r_cut**2:
                r = np.sqrt(r2) 
                
                s_over_r = sigma / r
                s_over_r2 = s_over_r * s_over_r
                s_over_r6 = s_over_r2 * s_over_r2 * s_over_r2
                s_over_r12 = s_over_r6 * s_over_r6
                
                E_pot += 4.0 * eps * (s_over_r12 - s_over_r6)
                
    return E_pot


@njit  
def pbc_distance(xi, xj, xlo, xhi):
    """Calculation of shortest distance via Minimum Image Convention."""
    L = xhi - xlo  # L

    rij = xj - xi  # Calculate raw distance

    # Apply Minimum Image Convention
    if abs(rij) > 0.5 * L:
        # A Numba-friendly way to apply MIC is using the round function: - chatgbt says thats very slow -->jonas: changed it to floor
        rij = rij - L * np.floor(rij / L)
        
    return rij


