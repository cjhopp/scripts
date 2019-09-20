#!/usr/bin/python
"""
Utilities for coordinate convertions at FS-B and SURF
"""

from pyproj import Proj

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
        return (self.orig_utm[0] - east, self.orig_utm[1] - north, point[2])