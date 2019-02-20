

# WARPING distance -- can theoretically handle noisy data better than DTW

# port from trajectory-distance package, with expansion from 2D to 3D spatial data

# note that this can easily be ported to C or GPU if needed


import numpy as np
from distance.euclidean import eucl_dist



def lcss(t0, t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """

    n0 = len(t0)+1
    n1 = len(t1)+1

    # An (m+1) times (n+1) matrix
    C=np.zeros((n0,n1))

    for i in range(1, n0):
        for j in range(1, n1):
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]

            if t0.shape[1] == 2:
                dist = eucl_dist(x0, y0, x1, y1)
            elif t0.shape[1] == 3:
                z0 = t0[i-1,2]
                z1 = t1[i-1,2]
                dist = eucl_dist(x0, y0, x1, y1, z0, z1)

            if dist<eps:
                C[i,j] = C[i-1,j-1] + 1
            else:
                C[i,j] = max(C[i,j-1], C[i-1,j])
    lcss = 1-float(C[n0-1,n1-1])/min(n0-1,n1-1)

    return lcss











