import numpy as np
from numba import njit
from parameters import dimensions, dimensions_task1, L

def create_particles(number_of_particles, n_steps, n_steps_equil, n_save, dimensions=dimensions):
    positions_equil = np.zeros((number_of_particles, dimensions, n_steps_equil // n_save))
    positions = np.zeros((number_of_particles, dimensions, n_steps // n_save))
    
    return positions, positions_equil


def set_initial_positions_2part(number_of_particles, n_steps, n_steps_equil, n_save, dimensions=dimensions):
    positions, positions_equil = create_particles(number_of_particles=number_of_particles,
                                                  n_steps=n_steps, n_steps_equil=n_steps_equil,
                                                  n_save=n_save, dimensions=dimensions)
    
    # Initialize only the FIRST time slice
    positions_equil[0,0,0] = L/4
    positions_equil[1,0,0] = L/2

    return positions_equil




def InitializeAtoms(random_seed, settings):
    # np.random.seed(settings.random_seed)    

    x = np.zeros(shape=(settings.N))
    y = np.zeros(shape=(settings.N))
    z = np.zeros(shape=(settings.N))
    vx = np.zeros(shape=(settings.N))
    vy = np.zeros(shape=(settings.N))
    vz = np.zeros(shape=(settings.N))

    n = 0
    i = 0
    while n < settings.N:
        x0 =  np.random.rand()*settings.L
        y0 =  np.random.rand()*settings.L
        z0 =  np.random.rand()*settings.L

        b = False
        for i in range(n):
            rijx = pbc(x[i], x0, 0, settings.L)
            rijy = pbc(y[i], y0, 0, settings.L)
            rijz = pbc(z[i], z0, 0, settings.L)
            
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            r = np.sqrt(r2)

            if r < settings.sig:
                b = True #reject placement
            
        if not b:
            x[n] = x0
            y[n] = y0
            z[n] = z0

            vx0 = 0.5 - np.random.rand()
            vy0 = 0.5 - np.random.rand()
            vz0 = 0.5 - np.random.rand()
            
            vx[n] = vx0
            vy[n] = vy0
            vz[n] = vz0

            n+= 1
        else:
            i +=1
            pass # tries same n again

        if i> settings.N**3:
            print('could not find enough positions, density to high')
            break

    
    return x, y, z, vx, vy, vz
