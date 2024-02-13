import numpy as np
import matplotlib.pyplot as plt

import nc_sim

width = 1
height = 1
H = np.random.rand(10,10)

X, Y = nc_sim.axons.place_neurons(width, height, H, rho=100)

W = nc_sim.axons.grow_W(width, height, X, Y, H=H)

nw = nc_sim.neurons.izhikevich.init_network(len(X), W, Tn=20e3, sigma=2.25)

nc_sim.neurons.izhikevich.run_nw(nw)

plt.plot(nw['spikes_T'], nw['spikes_Y'], '.')
plt.show()
