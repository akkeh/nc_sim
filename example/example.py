import numpy as np
import matplotlib.pyplot as plt

import nc_sim

width = 1   # width of culture
height = 1  # height of culture
H = np.random.rand(10,10)   # obstacles

# place neurons on plane:
X, Y = nc_sim.axons.place_neurons(width, height, H, rho=100)

# grow axons:
W = nc_sim.axons.grow_W(width, height, X, Y, H=H)

# initialise neurons:
nw = nc_sim.neurons.izhikevich.init_network(len(X), W, Tn=20e3, sigma=2.25)

# run dynamics:
nc_sim.neurons.izhikevich.run_nw(nw)

# plot:
plt.plot(nw['spikes_T'], nw['spikes_Y'], '.')
plt.show()
