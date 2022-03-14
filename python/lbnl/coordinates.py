#!/usr/bin/python
"""
Utilities for coordinate convertions at FS-B and SURF
"""

import numpy as np

from pyproj import Proj


def sdr_to_normal(strike, dip):
    dip_rad = np.deg2rad(dip)
    strike_rad = np.deg2rad(strike)
    # Normal to plane
    a = np.sin(dip_rad) * np.cos(strike_rad)  # East
    b = -np.sin(dip_rad) * np.sin(strike_rad)  # North
    c = np.cos(dip_rad)
    return np.array([a, b, c])


def cartesian_distance(pt1, pt2):
    """Helper distance calculation between two pts (x, y, z) in meters"""
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    dz = pt1[2] - pt2[2]
    return np.sqrt(dx**2 + dy**2 + dz**2)


class SURF_converter:
    """
    0, 0 for HMC:
    NAD27: 44.358982 -103.76546
    WGS84: 44.35894 -103.76604
    UTM Z13 WGS84: 4912481 N 598334 E
    """
    def __init__(self):
        # self.orig_utm = (598334.1035272244, 4912479.756701191)  # WGS84 UTM
        self.orig_utm = (598420.3842806489, 4912272.275375654)  # NAD27 UTM
        # Create the Proj for each
        # self.utm = Proj(init="EPSG:32613")  # WGS84UTM
        self.utm = Proj(init="EPSG:26713") # NAD27UTM

    def to_lonlat(self, point):
        """
        Take (y, x, z) point in HMC, rough convert to UTM, then convert to
        global coords.

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