import numpy as np
import matplotlib.pyplot as plt

from Plot import apply_style
from SaveToFile import load_hist
import os

apply_style()

def plot_multiple_hists(params):
    """
    params: list of (rho, epsilon) tuples
            corresponding to filenames: hist_{rho}_{epsilon}.txt
    """
    plt.figure()
    print(os.getcwd())
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # change the working directory to the file directory
    print(os.getcwd())
    for rho, epsilon in params:
        filename = f"hist_{rho}_{int(epsilon)}.txt"

        # Load
        hist, dr = load_hist(filename)

        # Construct bin centers
        bins = np.arange(len(hist)) * dr + dr/2

        # Plot
        plt.plot(bins, hist, label=f"ρ={rho}, ε={epsilon}")

    plt.xlabel(r"$r$")
    plt.ylabel(r"$g(r)$")
    # plt.title("Comparison of Histograms")
    # plt.ylim(0, 4)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
params = [ 
    (0.1, 1.0),
    (0.5, 1.0),
    (0.025, 1.0)
]

params2 = [
    (0.1, 10.0),
    (0.5, 10.0)]
# params = [(0.5, 1)]

plot_multiple_hists(params)
plot_multiple_hists(params2)
