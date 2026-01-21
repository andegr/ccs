import numpy as np
import numba
from numba import njit, prange, int32, float64
from parameters import MDSimulationParameters

@njit
def sample_zeta(n_particles, dimensions=2):
    zeta = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, dimensions))
    return zeta


@njit
def calc_discforce(position, r_disc, epsilon_disc, L):
    """
    Calculates the external disc confinement force acting on a particle.

    position : array([x, y])
    r_disc   : disc radius R
    epsilon_disc : confinement strength ε
    """

    # disc center (assumes global box length L)
    cx = 0.5 * L
    cy = 0.5 * L

    # displacement from center
    dx = position[0] - cx
    dy = position[1] - cy

    # radial distance
    r = np.sqrt(dx*dx + dy*dy)

    # distance to wall
    dr = r_disc - r

    # prefactor of the force
    pref = -9.0 * epsilon_disc / (dr**10)

    # radial unit vector
    fx = pref * dx / r
    fy = pref * dy / r

    return fx, fy




# Note: Parrallelization makes only sense for many particles, otherwise it is slower


@njit(parallel=True)
def Euler_Maruyama(positions, orientations,
                   dimensions, L, r_cut, eps, sigma,
                   dt, kB, T, Dt, Dr, v0, walls, pairwise, disc, r_disc, epsilon_disc):
    """
    One Euler–Maruyama step for free Active Brownian Particles (2D).
    """

    n_particles = positions.shape[0]

    new_positions = np.empty_like(positions)
    new_orientations = np.empty_like(orientations)

    # Remember: Effective propulsion speed v0 = beta * Dt * F

    zeta_r = sample_zeta(n_particles, dimensions)
    prefactor_r = np.sqrt(2.0 * Dt * dt)

    zeta_theta = np.random.randn(n_particles)
    prefactor_theta = np.sqrt(2.0 * Dr * dt)

    # pairwise forces computed once
    if pairwise:
        forces = force_pairwise_wca(positions, L=L, eps=eps, sigma=sigma, r_cut=r_cut, pbc=True)
    else:
        forces = np.zeros_like(positions)

    for i in prange(n_particles):

        if disc:
            fx, fy = calc_discforce(positions[i, :], r_disc, epsilon_disc, L)
            forces[i, 0] += fx
            forces[i, 1] += fy


        # orientation update
        theta_new = orientations[i] + prefactor_theta * zeta_theta[i]
        new_orientations[i] = theta_new % (2*np.pi)

        # propulsion direction
        nx = np.cos(theta_new)
        ny = np.sin(theta_new)

        # position update
        new_positions[i, 0] = positions[i, 0] + v0 * nx * dt + Dt * forces[i, 0] * dt + prefactor_r * zeta_r[i, 0] 
        new_positions[i, 1] = positions[i, 1] + v0 * ny * dt + Dt * forces[i, 1] * dt + prefactor_r * zeta_r[i, 1] 

        if walls:
            if (new_positions[i, 0] < 0) or (new_positions[i, 0] > L):
                new_positions[i, 0] = positions[i, 0]
                # not resetting the y positions as well is like realizing half a step
                # and I think it could cause unexpected behaviour
                new_positions[i, 1] = positions[i, 1]


            # #-----------other wall implementation (what physical interpretation???)----------
            # if (new_positions[i, 0] < 0):
            #     new_positions[i, 0] = 0
                

            # elif (new_positions[i, 0] > L):
            #     new_positions[i, 0] = L

            # #---------------------------------------------------


        # --- periodic boundary conditions ---
        # commented out for MSD calculation
        # new_positions[i, 0] %= L
        # new_positions[i, 1] %= L

    return new_positions, new_orientations





@njit(parallel=True)
def force_pairwise_wca(positions, L, eps, sigma, r_cut, pbc=True):
    """
    Compute WCA forces on each particle.
    positions: (N, 2)
    returns: forces (N, 2)
    """
    N = positions.shape[0]
    forces = np.zeros((N, 2), dtype=positions.dtype)

    rcut2 = r_cut * r_cut
    sig6  = sigma**6
    sig12 = sig6 * sig6

    for i in prange(N):
        fix = 0.0
        fiy = 0.0
        xi = positions[i, 0]
        yi = positions[i, 1]

        for j in range(N):
            if j != i:

                xij = xi - positions[j, 0]
                yij = yi - positions[j, 1]
                if pbc:
                    xij = pbc_distance(xi, positions[j, 0], 0, L)
                    yij = pbc_distance(yi, positions[j, 1], 0, L)

                r2 = xij*xij + yij*yij
                if r2 < rcut2:
                    # F_vec = 24 eps * (2*sigma^12/r^14 - sigma^6/r^8) * r_vec
                    inv_r2 = 1.0 / r2
                    inv_r6 = inv_r2 * inv_r2 * inv_r2
                    inv_r8  = inv_r6 * inv_r2
                    inv_r14 = inv_r8 * inv_r6

                    coeff = 24.0 * eps * (2.0 * sig12 * inv_r14 - sig6 * inv_r8)
                    fix += coeff * xij
                    fiy += coeff * yij

        forces[i, 0] = -fix
        forces[i, 1] = -fiy

    return forces



@njit
def pbc_distance(xi, xj, xlo, xhi):
    L = xhi - xlo
    rij = xj - xi
    rij -= L * np.rint(rij / L)
    return rij


