#!/usr/bin/env python

"""
Script to deal with checking picks/waveforms during template correlation
"""

import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
from glob import glob
from obspy import read, read_events
from obspy.core.event import ResourceIdentifier
from eqcorrscan.utils import plotting

temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/refined_picks/*'
temp_files = glob(temp_dir)

# Template dictionary keyed to event resource_id
template_dict = {}
for filename in temp_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[-1].rstrip('.mseed')
    uri = ResourceIdentifier(uri_name)
    template_dict[uri] = read(filename)

# Raw template dictionary keyed to event resource_id
raw_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/events_raw/*'
raw_files = glob(raw_dir)

raw_dict = {}
for filename in raw_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[-1].rstrip('.mseed')
    uri = ResourceIdentifier(uri_name)
    raw_dict[uri] = read(filename)

# Grab some catalog of interest
cat_list = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/corr_groups/*029*')
cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/qml/2015_nlloc_final_run02_group_refined.xml')

# Plotting with multi_event_singlechan


# Plot a template over raw data? Not sure this works correctly
rid = cat[0].resource_id
temp_st = template_dict[rid]
raw_st = raw_dict[rid]
raw_st.filter('bandpass', freqmin=1.0, freqmax=20)
times = []
for tr in raw_st:
    temp_tr_time = [p.time for p in cat[0].picks
                    if p.waveform_id.station_code == tr.stats.station and
                    p.waveform_id.channel_code == tr.stats.channel]
    if temp_tr_time:
        times.append(temp_tr_time[0])
plotting.detection_multiplot(raw_st, temp_st, times, plot_mode='single')
