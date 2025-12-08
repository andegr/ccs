import numpy as np
from SaveToFile import load_hist

import numpy as np
# from SaveToFile import load_hist # Assuming this is available

def calc_compressibility(params):
    """
    Calculates the isothermal compressibility (kappa_T) from the RDF g(r)
    using the pressure equation of state. Assumes k_B*T = 1.
    """
    kappas = np.zeros(len(params))

    for i, (rho, epsilon) in enumerate(params):
        
        filename = f"sheet7/hist_{rho}_{int(epsilon)}.txt" 
        
        try:
            hist, dr = load_hist(filename)
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping calculation for rho={rho}.")
            kappas[i] = np.nan
            continue # Skip to the next parameter set
            
        r_arr = np.arange(len(hist)) * dr + dr/2

        integrand_term = (hist - 1) * r_arr**2
        
        integral_term = 4 * np.pi * np.sum(integrand_term * dr)
        
        # 4. Final kappa_T calculation (assuming k_B*T = 1)
        kappas[i] = 1/rho + integral_term

    return kappas

# Example usage:
params = [
    (0.5, 1.0),
    (0.5, 10.0),
    (0.1, 1.0),
    (0.1, 10.0),
    (0.025, 1.0)]

kappas = calc_compressibility(params)

print(kappas)