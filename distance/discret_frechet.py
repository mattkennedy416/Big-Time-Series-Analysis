


import numpy as np
from distance.euclidean import eucl_dist


def discret_frechet(t0, t1):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q

    Parameters
    ----------
    param t0 : px2 numpy_array, Trajectory t0
    param t1 : qx2 numpy_array, Trajectory t1

    Returns
    -------
    frech : float, the discret frechet distance between trajectories t0 and t1
    """
    n0 = len(t0)+1
    n1 = len(t1)+1
    C=np.zeros((n0,n1))

    C[1:,0]=np.inf
    C[0,1:]=np.inf
    for i in range(1, n0):
        for j in range(1, n1):
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            C[i,j]=max(eucl_dist(x0,y0,x1,y1),min(min(C[i,j-1],C[i-1,j-1]),C[i-1,j]))
    dfr = C[n0-1,n1-1]
    return dfr