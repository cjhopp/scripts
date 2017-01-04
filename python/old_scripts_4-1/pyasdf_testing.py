#!/usr/bin/env python

import obspy
import pyasdf
from glob import glob
import os

station_files = glob('/home/chet/data/GeoNet_catalog/stations/*XML.xml')
event_files = glob('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotokawa/part_2/*.xml')
wave_files = ''

#Use as context manager to ensure file is always closed
with pyasdf.ASDFDataSet("/home/chet/data/mrp_data/pyasdf/mrp_rotokawa.h5") as ds:
    #Add station files
    for afile in station_files:
        ds.add_stationxml(afile)
    #Add events
    ds.add_quakeml('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotokawa/part_2/rot_pt2_QML.xml')
