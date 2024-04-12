import numpy as np

def place_neurons(width, height, H=[], **kwargs):
    width       = width     # culture width (mm)
    height      = height    # culture height (mm)
    H           = H         # obstacles MxN grid of numbers:
                            #   0   : no obstacle; 
                            #   h>0 : obstacle of height h mm; 
                            #   h<0 : no neurons or axons can grow here

    # default parameters:
    rho     = 100       # neuron density (neurons / mm^2)
    r_soma  = 7.5e-3    # soma radius (mm)
    
    for argn, argv in kwargs.items():
        match argn:
            case 'rho':
                rho = float(argv)
            case 'r_soma':
                r_soma = float(argv)

    # some derived parameters:
    M = int(height*width*rho)   # number of neurons  
    Hm,Hn = np.shape(H) if np.size(H) > 0 else (1,1)  # size of obstacles (PDMS) grid
    cell_height = height/Hm
    cell_width = width/Hn

    # place neurons non overlapping on area:
    X,Y = np.zeros((2,M))
    X[0] = np.random.rand()*width
    Y[0] = np.random.rand()*height
    for i in range(1,M):
        X[i] = np.random.rand()*width
        Y[i] = np.random.rand()*height
        r,c = get_cell(X[i], Y[i], cell_width, cell_height)
        while np.any(np.sqrt(np.power(X[:i]-X[i],2)+np.power(Y[:i]-Y[i],2)) < r_soma) or get_H(r,c,H) < 0:
            X[i] = np.random.rand()*width
            Y[i] = np.random.rand()*height
            r,c = get_cell(X[i], Y[i], cell_width, cell_height)

    return X,Y

def grow_W(width, height, X, Y, H=[], **kwargs):
    width       = width     # culture width (mm)
    height      = height    # culture height (mm)
    H           = H         # obstacles MxN grid of numbers:
                            #   0   : no obstacle; 
                            #   h>0 : obstacle of height h mm; 
                            #   h<0 : no neurons or axons can grow here

    # default arguments:
    L       = 1.00  # average axon length (mm)
    Dl      = 1e-3  # axon segment length (mm)
    phi_sd  = 0.1   # axon random walk std

    r_dendrite_mu   = 150e-3    # denrite radius mean (mm)
    r_dendrite_sd   = 20e-3     # denrite radius std

    # read optional arguments:
    for argn, argv in kwargs.items():
        match argn:
            case 'L':
                L = float(argv)
            case 'Dl':
                Dl = float(argv)
            case 'phi_sd':
                phi_sd = float(argv)
            case 'r_dendrite_mu':
                r_dendrite_mu = float(argv)
            case 'r_dendrite_sd':
                r_dendrite_sd = float(argv)

    # some derived parameters:
    M = len(X)              # number of neurons
    Hm,Hn = np.shape(H) if np.size(H) > 0 else (1,1)  # size of obstacles (PDMS) grid
    cell_height = height/Hm
    cell_width = width/Hn

    # init:
    W = np.zeros((M,M)) # resulting adjacency matrix

    r_dendrite = r_dendrite_mu + r_dendrite_sd*np.random.normal(0,1,M)

    Ln = L*np.sqrt(-2*np.log(1-np.random.rand(M)))  # axon lengths
    Nl = int(np.max(Ln)/Dl)

    Xi = np.copy(X)
    Yi = np.copy(Y)
    phi = 2*np.pi*np.random.rand(M)

    # main loop:
    for n in np.arange(1,Nl):
        # determine axon growth direction:
        Dx = np.multiply(Dl*np.cos(phi), (n*Dl)<Ln)
        Dy = np.multiply(Dl*np.sin(phi), (n*Dl)<Ln)

        # temporarily update axon cone:
        Xj = Xi+Dx
        Yj = Yi+Dy

        for i in np.arange(M)[np.array((n*Dl)<Ln)]:
            # check for border crossings:
            _, Xj[i], Yj[i], phi[i], Ln[i] = check_crossing(Xi[i], Yi[i], Xj[i], Yj[i], H, phi[i], Ln[i], Dl, cell_width, cell_height)

            # determine connections:
            P = np.sqrt(np.power(Xj[i]-X,2) + np.power(Yj[i]-Y,2)) < r_dendrite
            P[i] = 0    # no self-connections
            for j in np.arange(M)[P==1]:    # for all neurons for which axon is close
                # check whether dendrites cross border:
                crossed, _, _, _, _ = check_crossing(X[j], Y[j], Xj[i], Yj[i], H, phi[i], Ln[i], Dl, cell_width, cell_height)
                if crossed:
                    W[i,j] = 1

        # update axon cone positions:
        Xi = Xj
        Yi = Yj

        # random deviation for next step:
        phi += phi_sd*np.random.normal(0, 1, M)

    return W

def get_cell(x, y, cell_width, cell_height):
    row = int(y/cell_height)
    col = int(x/cell_width)
    return row,col

def get_H(row, col, H):
    if len(H) == 0: return 0

    M,N = np.shape(H)
    
    row = 0 if row < 0 else ( M-1 if row >= M else row )
    col = 0 if col < 0 else ( N-1 if col >= N else col )

    return H[row,col]

def get_P(H_i, H_j):
    # values from Hernandez-Navarro thesis
    h = np.array([ 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p01 = np.array([ 1, 0.0007, 0.00045, 0.00035, 0.00030, 0.00025, 0.00015, 0.00002, 0 ])
    p10 = np.array([ 1, 0.0066, 0.0033, 0.0033, 0.0033, 0.0033, 0.00200, 0.00050, 0 ]) 
    Dh = H_j-H_i
    if Dh > 0:
        return np.interp(Dh, h, p01)
    else:
        return np.interp(abs(Dh), h, p10)

def check_crossing(xi, yi, xj, yj, H, phi, Ln, Dl, cell_width, cell_height):
    crossed = True
    if len(H) > 0:
        re_check = True
        while re_check:
            re_check = False
            row_i, col_i = get_cell(xi, yi, cell_width, cell_height)
            row_j, col_j = get_cell(xj, yj, cell_width, cell_height)

            H_i = get_H(row_i, col_i, H)
            H_j = get_H(row_j, col_j, H)

            P_cross = get_P(H_i, H_j)
            if P_cross < 1: # so, maybe we do not cross
                xborder = -1 if col_i == col_j else np.max([col_i, col_j])*cell_width
                yborder = -1 if row_i == row_j else np.max([row_i, row_j])*cell_height

                A = (xi, yi)
                B = (xj, yj)
                if xborder >= 0 and yborder >= 0:
                    # we are crossing 2 borders, life is hard
                    Px = (xborder, -1000*cell_height)
                    Qx = (xborder, 1000*cell_height)
                    Py = (-1000*cell_width, yborder)
                    Qy = (1000*cell_width, yborder)
                    P,Q = (Py, Qy) if distance(A, Py, Qy) < distance(A, Px, Qx) else (Px, Qx)
                elif xborder >= 0:
                    P = (xborder, -1000*cell_height)
                    Q = (xborder, 1000*cell_height)
                elif yborder >= 0:
                    P = (-1000*cell_width, yborder)
                    Q = (1000*cell_width, yborder)
                else: pass  # should not happen though

                angle_AB_on_PQ = get_angle(A, B, P, Q)
                if ( abs(angle_AB_on_PQ) <= np.pi/6.0 ) or ( np.random.rand() < (1-P_cross) ):
                    # deflect axon:
                    phi = get_slope(P, Q) if angle_AB_on_PQ < 0 else get_slope(Q, P)
                    xj = xi + Dl*np.cos(phi)
                    yj = yi + Dl*np.sin(phi)
                    re_check = True
                    crossed = False
                else:
                    # axon crosses the obstacle:
                    Ln -= abs(H_j-H_i)
    return crossed, xj, yj, phi, Ln

def distance(A,P,Q):
    num = (Q[0]-P[0])*(P[1]-A[1]) - (P[0]-A[0])*(Q[1]-P[1])
    den = np.sqrt((Q[0]-P[0])**2+(Q[1]-P[1])**2)
    return num/den

def get_slope(A,B):
    return np.arctan2(B[1]-A[1],B[0]-A[0])

def get_angle(A,B,P,Q):
    ''' angle of AB w. resp of PQ '''
    return np.angle(np.exp(1j*(get_slope(A,B)-get_slope(P,Q))))

