hist, bins = np.histogram(positions[0], bins=100, density=True)
hist *= n_particles
bin_centers = 0.5*(bins[:-1] + bins[1:])