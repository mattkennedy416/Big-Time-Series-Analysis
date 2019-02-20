


from numpy import sin, cos, arctan2, sqrt, pi


# port from trajectory-distance package


rad = pi/180
R = 6378137

def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)

    Parameters
    ----------
    param lon1: float, longitude of the first point
    param lat1: float, latitude of the first point
    param lon2: float, longitude of the second point
    param lat2: float, latitude of the second point

    Returns
    -------
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """


    dLat = rad*(lat2 - lat1)
    dLon = rad*(lon2 - lon1)
    a = (sin(dLat / 2) * sin(dLat / 2) +
    cos(rad*(lat1)) * cos(rad*(lat2)) *
    sin(dLon / 2) * sin(dLon / 2))
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    d = R * c
    return d
