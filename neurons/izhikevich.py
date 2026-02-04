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

def init_network(N, A, **kwargs):
    params = {
        'P_connect' : 0.5,
        'EI_ratio'  : 0.8,
        'gE'        : 6,    # excitatory strength
        'gI'        : -12,  # inhibitory strength

        # Izhikevich parameters:
        'alpha'     : 0.04,
        'beta'      : 5,
        'gamma'     : 140,
        'a'         : 0.02, # u-timescale
        'b'         : 0.2,  # u v-sensitivity
        'v0'        : -65,  # v-reset (absolute value)
        'vc'        : 30,   # threshold
        'delta_u'   : 6.5,  # u-`reset' (add. factor)

        # synapse dynamics:
        'tauE'  : 10,   # exc. time-scale (decay)
        'tauI'  : 10,   # inh. time-scale (decay)
        'beta_R': 0.8,  # R-`reset' (mult. factor)
        'tauR'  : 8e3,  # synaptic depression recovery time-scale

        # noise driving:
        'sigma' : 2.00,   # noise magnitude

        # sim. parameters:
        'Dt'    : 1e0,  # time-step size (ms)
        'Tn'    : 1e3,  # total sim. time (ms)
    }
    
    # update params:
    params.update(kwargs)
    params['Nt'] = int(params['Tn']/params['Dt'])
    params['gW'] = np.sqrt(2*params['sigma']/params['Dt'])

    # set seed:
    if 'seed' in params: np.random.seed(params['seed'])

    Ne = int(N*params['EI_ratio'])
    Ni = N-Ne
    B = np.multiply(A, np.random.rand(N,N) > (1-params['P_connect']))
    W = np.multiply(B, params['gE']*np.random.normal(1, 0.1, (N,N)))
    W[:,Ne:] = np.multiply(B[:,Ne:], params['gI']*np.random.normal(1, 0.1, (N, Ni)))

    v = np.random.rand(N) * (params['vc']-params['v0']) + params['v0']
    u = params['b']*v

    P = np.random.rand(N)
    R = np.random.rand(N)

    nw = {
        'N' : N,
        'Ne': Ne,
        'Ni': Ni,

        'A' : A,
        'W' : W,

        'v' : v,
        'u' : u,
        'P' : P,
        'R' : R,

        'dv': v*0,
        'du': u*0,
        'dP': P*0,
        'dR': R*0,

        'Inp'   : v*0,
        'eta'   : v*0,

        'S'         : np.mat(np.zeros((N,1))),
        'spike_T'   : [],
        'spike_Y'   : [],

        'params' : params
    }
    return nw

def simulation_step(n, nw, Iext=None):
    v = nw['v']
    u = nw['u']
    P = nw['P']
    R = nw['R']
    dv = nw['dv']
    du = nw['du']
    dP = nw['dP']
    dR = nw['dR']
    Inp = nw['Inp']
    eta = nw['eta']

    pars = nw['params']

    Ne  = nw['Ne']
    Dt  = pars['Dt']

    Inp[:]  = np.dot(nw['W'], P)    # synaptic input
    eta[:]  = pars['gW']*np.random.normal(0,1,nw['N'])   # noise drive
    if Iext is not None: Inp[:] + Iext
    
    du[:]   = pars['a']*(pars['b']*v - u)
    dP[:Ne] = -P[:Ne]/pars['tauE']
    dP[Ne:] = -P[Ne:]/pars['tauI']
    dR[:]   = (1-R)/pars['tauR']

    dv[:]   = pars['alpha']*np.power(v,2) + pars['beta']*v + pars['gamma'] - u
    v[:]    = v + 0.5*(dv + Inp + eta)*Dt
    dv[:]   = pars['alpha']*np.power(v,2) + pars['beta']*v + pars['gamma'] - u
    v[:]    = v + 0.5*(dv + Inp + eta)*Dt

    u[:]    = u + du*Dt
    P[:]    = P + dP*Dt
    R[:]    = R + dR*Dt

    nw['S'][:] = 0
    for i in np.arange(len(v))[v>pars['vc']]:
        v[i] = pars['v0']
        u[i] = u[i] + pars['delta_u']
        P[i] = P[i] + R[i]
        R[i] = (1-pars['beta_R'])*R[i]
        nw['spike_T'].append(n*Dt)
        nw['spike_Y'].append(i)
        nw['S'][i] = 1

    return nw   # do not capture, nw is updated by reference

def run_nw(nw, Iext=None):
    Nt = nw['params']['Nt']
    for n in np.arange(1,Nt):
        iext = Iext[:,n] if Iext is not None else None
        simulation_step(n, nw, iext)

    
