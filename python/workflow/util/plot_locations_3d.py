#!/usr/bin/env python

"""
Plotting events in 3D with stations and wells
"""
from __future__ import division

import sys
sys.path.insert(0, '/home/chet/EQcorrscan')

import pandas as pd
from eqcorrscan.utils import plotting
import mpl_toolkits.basemap.pyproj as pyproj
from obspy import read_inventory, read_events
from glob import glob

# Wells first
df = pd.read_csv('/home/chet/data/mrp_data/well_data/locations/RK_E-N_Survey.csv')
wgs84 = pyproj.Proj("+init=EPSG:4326")
nzmg = pyproj.Proj("+init=EPSG:27200")
well_dict = {}
for index, row in df.iterrows():
    lon, lat = pyproj.transform(nzmg, wgs84, row[2], row[3])
    if row[0] not in well_dict:
        well_dict[row[0]] = [(lat, lon, (row[4] / 1000))]
    else:
        well_dict[row[0]].append((lat, lon, (row[4] / 1000)))
# Read catalog
# cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/qml/2015_nlloc_final_run02_group_refined.xml')
# Read inventory
files = glob('/home/chet/data/GeoNet_catalog/stations/station_xml/*')
for filename in files:
    if not 'inv' in locals():
        inv = read_inventory(filename)
    else:
        inv += read_inventory(filename)

fig = plotting.obspy_3d_plot(inv, hypoDD_cat)
ax = fig.gca()
for well, points in well_dict.iteritems():
    lats, lons, dps = zip(*points)
    ax.plot(lats, lons, dps)
