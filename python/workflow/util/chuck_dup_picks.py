#/usr/bin/env python

"""
Script to search a obspy.Catalog for duplicated picks and remove them
"""

import pyasdf
from obspy import read_events
test_name = '/home/chet/data/mrp_data/pyasdf/pyasdf_testing.h5'
asdf_name = '/media/chet/rotnga_data/pyasdf/mrp_rotnga.h5'
raw_qml = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga/rotnga_qml_noduppicks.xml'
#Read pyasdf events to Catalog
with pyasdf.ASDFDataSet(asdf_name) as ds:
    cat = ds.events
#OR read qml straight to Catalog
cat = read_events(raw_qml)

#For each event, loop over picks. Save new picks, discard any duplicates
for event in cat:
    unique_ids = []
    unique_picks = []
    for pick in event.picks:
        if pick.resource_id not in unique_ids:
            unique_ids.append(pick.resource_id)
            #Eliminate annoying comment in S picks
            if len(pick.comments) > 0:
                pick.comments = []
            unique_picks.append(pick)
    event.picks = unique_picks
#Eliminate duplicate amplitudes
for event in cat:
    unique_ids = []
    unique_amps = []
    for amp in event.amplitudes:
        if amp.resource_id not in unique_ids:
            unique_ids.append(amp.resource_id)
            unique_amps.append(amp)
    event.amplitudes = unique_amps
#Eliminate duplicate station magnitudes
for event in cat:
    unique_ids = []
    unique_sta_mags = []
    for sta_mag in event.station_magnitudes:
        if sta_mag.resource_id not in unique_ids:
            unique_ids.append(sta_mag.resource_id)
            unique_sta_mags.append(sta_mag)
    event.station_magnitudes = unique_sta_mags
#Eliminate duplicate magnitudes
for event in cat:
    unique_ids = []
    unique_mags = []
    for mag in event.magnitudes:
        if mag.resource_id not in unique_ids:
            unique_ids.append(mag.resource_id)
            unique_mags.append(mag)
    event.magnitudes = unique_mags

out_file_test = '/home/chet/data/mrp_data/testing/quakeml/test_no_dups.xml'
out_file = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga/final_cat/rotnga_qml_nodupsATALL.xml'
cat.write(out_file, format="QuakeML")
