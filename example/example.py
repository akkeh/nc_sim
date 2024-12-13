'''
    nc_sim: neuronal culture growth and activity simulation in environments with obstacles
    Please cite: ...
    Copyright (C) 2024 Akke Mats Houben (akke@akkehouben.net)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

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
