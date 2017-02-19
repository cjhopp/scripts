#!/usr/bin/env python
"""
Use this to compare individual event template detections with the \
corresponding subspace detectors
"""
import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
import pandas as pd
from datetime import datetime
from eqcorrscan.utils import plotting
from obspy import UTCDateTime, read_events
from glob import glob
# Locate catalog of events clustered in space
corr_groups = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/corr_groups/1_sec_temps/spacegrp_06*')
# Create two dataframes for each set of detections
df_sub = pd.read_csv('/home/chet/data/detections/2015_corr_groups/2015_all_det_corrgrps.csv', header=None)
df_all = pd.read_csv('/home/chet/data/detections/2015_alltemps_1sec/2015_all_det_1sec.csv', header=None)
df = pd.concat([df_sub, df_all])
det_dict = {}
for index, row in df.iterrows():
    if row[0] in det_dict:
        if UTCDateTime(row[1]) < UTCDateTime(2015, 11, 20) and row[4] > 5:
            det_dict[row[0]].append(UTCDateTime(row[1]).datetime)
    else:
        if UTCDateTime(row[1]) < UTCDateTime(2015, 11, 20) and row[4] > 5:
            det_dict[row[0]] = [UTCDateTime(row[1]).datetime]
for grp in corr_groups:
    temp_names = []
    temp_times = []
    cat = read_events(grp)
    if len(cat) > 1:
        sub_dets = grp.split('/')[-1].rstrip('.xml')
        subs = [sub_dets + '_2n', sub_dets + '_1st']
        plot_file = '/media/chet/hdd/seismic/NZ/figs/cum_det_plots/' +\
                    sub_dets + '.png'
        for sub in subs:
            temp_names.append(sub)
            temp_times.append(det_dict[sub])
        for ev in cat:
            ev_name = str(ev.resource_id).split('/')[-1]
            temp_names.append(ev_name)
            temp_times.append(det_dict[ev_name + '_1sec'])
        plotting.cumulative_detections(temp_times, temp_names, show=True,
                                       save=False, savefile=plot_file)
