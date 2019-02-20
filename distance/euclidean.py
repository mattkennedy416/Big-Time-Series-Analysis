
import numpy as np

# port from trajectory-distance package

def eucl_dist(x1, y1, x2, y2, z1=None, z2=None):
    """
    Usage
    -----
    L2-norm between point (x1,y1) and (x2,y2)

    Parameters
    ----------
    param x1 : float
    param y1 : float
    param x2 : float
    param y2 : float

    Returns
    -------
    dist : float
           L2-norm between (x1,y1) and (x2,y2)
    """
    dx = (x1 - x2)
    dy = (y1 - y2)

    if z1 is None or z2 is None:
        d=np.sqrt(dx*dx+dy*dy)
    else:
        dz = (z1 - z2)
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
    return d

