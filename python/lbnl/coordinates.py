#!/usr/bin/python
"""
Utilities for coordinate convertions at FS-B and SURF
"""

import numpy as np

from pyproj import Proj


def cartesian_distance(pt1, pt2):
    """Helper distance calculation between two pts (x, y, z) in meters"""
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    dz = pt1[2] - pt2[2]
    return np.sqrt(dx**2 + dy**2 + dz**2)


class SURF_converter:
    def __init__(self):
        self.orig_utm = (598420.3842806489, 4912272.275375654)
        # Create the Proj for each
        self.utm = Proj(init="EPSG:26713")

    def to_lonlat(self, point):
        """
        Take (y, x, z) point in HMC, rough convert to UTM, then convert to
        cartesian coords.

        :param point: Tuple of (easting, northing, elevation)
        :return: tuple of (lon, lat, elev)
        """
        pt_utm = (self.orig_utm[0] + point[0], self.orig_utm[1] + point[1])
        lon, lat = self.utm(pt_utm[0], pt_utm[1], inverse=True)
        return (lon, lat, point[2])

    def to_HMC(self, point):
        """
        Inverse operation from above

        :param point: tuple of (lon, lat, elev)
        :return: tuple of (easting, northing, elev)
        """
        east, north = self.utm(point[0], point[1])
        return (east - self.orig_utm[0], north - self.orig_utm[1], point[2])


class FSB_converter:
    def __init__(self):
        self.utm = Proj(init='EPSG:2056')

    def to_lonlat(self, point):
        """
        Take y, x, z point on the ch1903+ grid and return lon lat

        :param point: (easting, northing, elevation)
        :return: (lon, lat, elevation)
        """
        lon, lat = self.utm(point[0], point[1], inverse=True)
        return (lon, lat, point[2])

    def to_ch1903(self, point):
        """
        Inverse of above. We don't touch elevations in either operation

        :param point: (lon, lat, elevation)
        :return: (easting, northing, elevation)
        """
        east, north = self.utm(point[0], point[1])
        return (east, north, point[2])