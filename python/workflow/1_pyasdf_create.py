#/usr/bin/env python

"""
Create pyasdf file
"""

import os
import pyasdf
import fnmatch
from glob import glob

asdf_name = '/home/chet/data/mrp_data/pyasdf/pyasdf_testing.h5'
asdf_name = '/media/chet/rotnga_data/pyasdf/'
wav_dir = '/media/chet/rotnga_data/waveform_data/2015'
sta_dir = '/home/chet/data/GeoNet_catalog/stations/station_xml/*'

with pyasdf.ASDFDataSet(asdf_name) as ds:
    # patterns = ['*ALRZ*2015.001', '*ARAZ*2015.001', '*HRRZ*2015.001',
    #             '*PRRZ*2015.001', '*WPRZ*2015.001', '*THQ2*2015.001']
    wav_files = []
    # for pat in patterns:
    for root, dirnames, filenames in os.walk(wav_dir):
        for filename in fnmatch.filter(filenames, '*'):
            wav_files.append(os.path.join(root, filename))

    for _i, filename in enumerate(wav_files):
        print("Adding mseed file %i of %i..." % (_i+1, len(wav_files)))
        st = read(filename)
        #Add waveforms
        ds.add_waveforms(st, tag="raw_recording")

#Add events (after deleting them, if you so choose)
with pyasdf.ASDFDataSet(asdf_name) as ds:
    # del ds.events
    ds.add_quakeml('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga/final_cat/rotnga_qml_nodupsATALL.xml')

with pyasdf.ASDFDataSet(asdf_name) as ds:
    files = glob(sta_dir)
    for filename in files:
        ds.add_stationxml(filename)
