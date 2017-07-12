#!/usr/bin/env python

"""
Write inventory object to NLLOC station format
"""

from glob import glob
from obspy import read_inventory

#Read inv from raw StationXMLs
files = glob('/home/chet/data/GeoNet_catalog/stations/station_xml/*')
for filename in files:
    if not 'inv' in locals():
        inv = read_inventory(filename)
    else:
        inv += (read_inventory(filename))
#Loop over each station and write the NLLOC accepted format
import csv
with open('/home/chet/data/mrp_data/inventory/' +
          'rotnga_2015_inv_nodepth.csv', 'wb') as f:
    csvwriter = csv.writer(f, delimiter=' ', escapechar=' ',
                           quoting=csv.QUOTE_NONE)
    for net in inv:
        tmp_sta = net.stations[0]
        name = str(tmp_sta.code)
        lat = str(tmp_sta.latitude)
        lon = str(tmp_sta.longitude)
        #Elevation in km...whatever
        elev = tmp_sta.elevation / 1000
        # Account for borehole depths
        for chan in tmp_sta.channels:
            if chan.depth != 0.0:
                depth = chan.depth / 1000
        if 'depth' in locals():
            elev = str(elev - depth)
            del depth
        else:
            elev = str(elev)
        #Not entirely sure why adding the ' ' to this produced 3 spaces...
        csvwriter.writerow(['GTSRCE', name, 'LATLON', lat, lon, '0',
                            ' ', elev])
